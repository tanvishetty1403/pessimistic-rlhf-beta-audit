
import numpy as np
import pytest

from src import pessimism as P
from src.rloo_reward import (
    compute_ensemble_reward,
    compute_mean_reward,
)


def test_k5_reward_matches_existing_gaussian_formula():
    scores = np.array([
        [0.2, 0.4, 0.1, 0.7, 0.3],
        [1.0, 0.8, 1.2, 0.9, 1.1],
    ])

    result = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        r_center=0.0,
        r_scale=1.0,
        smoke_test_placeholder=True,
    )

    expected = P.r_gauss(scores, P.beta_rob(1.0, 0.5))

    np.testing.assert_allclose(result.pessimistic_reward, expected)
    np.testing.assert_allclose(result.mean_reward, scores.mean(axis=-1))
    np.testing.assert_allclose(
        result.variance, scores.var(axis=-1, ddof=1)
    )
    np.testing.assert_allclose(
        result.penalty,
        result.mean_reward - result.pessimistic_reward,
    )


def test_tied_lambda_identity():
    scores = np.array([0.2, 0.4, 0.1, 0.7, 0.3])
    c = 1.0
    beta_policy = 0.5

    result = compute_ensemble_reward(
        scores,
        beta_policy=beta_policy,
        c=c,
        smoke_test_placeholder=True,
    )

    expected = P.r_uwo(scores, P.lam_tied(c, beta_policy))
    np.testing.assert_allclose(result.pessimistic_reward, expected)


def test_zero_calibration_scale_equals_mean():
    scores = np.array([0.2, 0.4, 0.1, 0.7, 0.3])

    result = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        calibration_scale=0.0,
        smoke_test_placeholder=True,
    )

    np.testing.assert_allclose(result.pessimistic_reward, P.r_mean(scores))
    np.testing.assert_allclose(result.penalty, 0.0)


def test_penalty_is_nonnegative():
    scores = np.array([
        [0.2, 0.4, 0.1, 0.7, 0.3],
        [1.0, 0.8, 1.2, 0.9, 1.1],
    ])

    result = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        smoke_test_placeholder=True,
    )

    assert np.all(result.penalty >= -1e-12)


def test_larger_c_reduces_penalty():
    scores = np.array([
        [0.2, 0.4, 0.1, 0.7, 0.3],
        [1.0, 0.8, 1.2, 0.9, 1.1],
    ])

    small_c = compute_ensemble_reward(
        scores, beta_policy=0.5, c=0.5,
        smoke_test_placeholder=True,
    )
    large_c = compute_ensemble_reward(
        scores, beta_policy=0.5, c=2.0,
        smoke_test_placeholder=True,
    )

    assert np.all(large_c.penalty <= small_c.penalty + 1e-12)


def test_penalty_scales_linearly_with_calibration_factor():
    scores = np.array([0.2, 0.4, 0.1, 0.7, 0.3])

    unscaled = compute_ensemble_reward(
        scores, beta_policy=0.5, calibration_scale=1.0,
        smoke_test_placeholder=True,
    )
    doubled = compute_ensemble_reward(
        scores, beta_policy=0.5, calibration_scale=2.0,
        smoke_test_placeholder=True,
    )

    np.testing.assert_allclose(doubled.penalty, 2.0 * unscaled.penalty)


def test_batch_shapes_are_preserved():
    scores = np.array([
        [0.2, 0.4, 0.1, 0.7, 0.3],
        [1.0, 0.8, 1.2, 0.9, 1.1],
        [0.1, 0.3, 0.2, 0.4, 0.5],
    ])

    result = compute_ensemble_reward(
        scores, beta_policy=0.5,
        smoke_test_placeholder=True,
    )

    assert result.member_scores.shape == (3, 5)
    assert result.mean_reward.shape == (3,)
    assert result.variance.shape == (3,)
    assert result.pessimistic_reward.shape == (3,)
    assert result.penalty.shape == (3,)


def test_frozen_standardization_is_affine_invariant():
    scores = np.array([
        [0.2, 0.4, 0.1, 0.7, 0.3],
        [1.0, 0.8, 1.2, 0.9, 1.1],
    ])

    center = 0.25
    norm_scale = 0.8

    original = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        r_center=center,
        r_scale=norm_scale,
    )

    # Transform raw scores and their frozen normalization parameters
    # consistently. The standardized scores must remain unchanged.
    a = 3.0
    b = -1.7
    transformed = a * scores + b

    equivalent = compute_ensemble_reward(
        transformed,
        beta_policy=0.5,
        r_center=a * center + b,
        r_scale=a * norm_scale,
    )

    np.testing.assert_allclose(
        original.member_scores, equivalent.member_scores
    )
    np.testing.assert_allclose(
        original.pessimistic_reward, equivalent.pessimistic_reward
    )
    np.testing.assert_allclose(original.penalty, equivalent.penalty)


def test_mean_baseline():
    scores = np.array([
        [0.2, 0.4, 0.1, 0.7, 0.3],
        [1.0, 0.8, 1.2, 0.9, 1.1],
    ])

    np.testing.assert_allclose(
        compute_mean_reward(scores), scores.mean(axis=-1)
    )


def test_rejects_wrong_ensemble_size():
    with pytest.raises(ValueError, match="Expected 5"):
        compute_ensemble_reward(
            [0.1, 0.2, 0.3], beta_policy=0.5
        )


def test_rejects_invalid_beta():
    with pytest.raises(ValueError, match="beta_policy"):
        compute_ensemble_reward(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            beta_policy=0.0,
        )


def test_rejects_invalid_normalization():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    with pytest.raises(ValueError, match="r_scale"):
        compute_ensemble_reward(
            scores, beta_policy=0.5, r_scale=0.0
        )


def test_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        compute_ensemble_reward(
            [0.1, 0.2, np.nan, 0.4, 0.5],
            beta_policy=0.5,
        )


def test_rejects_placeholder_flag_with_other_values():
    with pytest.raises(ValueError, match="placeholder"):
        compute_ensemble_reward(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            beta_policy=0.5,
            r_center=0.2,
            smoke_test_placeholder=True,
        )
