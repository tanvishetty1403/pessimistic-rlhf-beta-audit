
import json

import numpy as np
import pytest

from src import pessimism as P
from src.rloo_reward import (
    compute_ensemble_reward,
    compute_mean_reward,
    compute_std_stats,
    load_std_stats,
)


def test_reward_uses_manually_standardized_scores():
    raw = np.array([2.0, 4.0, 3.0, 7.0, 5.0])
    center = 3.5
    norm_scale = 2.0
    beta_policy = 0.7
    c = 1.3
    calibration_scale = 0.8

    std = (raw - center) / norm_scale
    result = compute_ensemble_reward(
        raw,
        beta_policy=beta_policy,
        c=c,
        calibration_scale=calibration_scale,
        r_center=center,
        r_scale=norm_scale,
        ensemble_id="test-k5",
        noise_condition="noise_off",
    )

    expected_mean = std.mean()
    expected_var = std.var(ddof=1)
    expected_reward = expected_mean - (
        calibration_scale * expected_var
        / (2.0 * c * beta_policy)
    )

    np.testing.assert_allclose(result.raw_scores, raw)
    np.testing.assert_allclose(result.std_scores, std)
    np.testing.assert_allclose(result.mean_reward, expected_mean)
    np.testing.assert_allclose(result.variance, expected_var)
    np.testing.assert_allclose(
        result.pessimistic_reward, expected_reward
    )
    np.testing.assert_allclose(
        result.penalty, expected_mean - expected_reward
    )

    assert result.c == c
    assert result.calibration_scale == calibration_scale
    assert result.ensemble_id == "test-k5"
    assert result.noise_condition == "noise_off"


def test_gaussian_approximation_matches_exact_reward():
    """Independent numerical check using a large Gaussian ensemble."""
    rng = np.random.default_rng(2026)
    beta = 1.2
    scores = rng.normal(loc=0.0, scale=0.35, size=2000)

    exact = P.r_exact(scores, beta)
    approximate = P.r_gauss(scores, beta)

    assert abs(approximate - exact) < 5e-3


def test_reference_stats_standardize_ensemble_means():
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(400, 5))
    reference += np.linspace(-2.0, 2.0, 400)[:, None]

    stats = compute_std_stats(reference)
    ensemble_means = reference.mean(axis=-1)
    standardized_means = (
        ensemble_means - stats["r_center"]
    ) / stats["r_scale"]

    assert abs(standardized_means.mean()) < 1e-12
    np.testing.assert_allclose(
        standardized_means.std(ddof=0), 1.0, atol=1e-12
    )


def test_compute_std_stats_matches_global_convention():
    reference = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0, 8.0, 9.0],
    ])

    means = reference.mean(axis=-1)
    stats = compute_std_stats(reference)

    np.testing.assert_allclose(stats["r_center"], means.mean())
    np.testing.assert_allclose(stats["r_scale"], means.std(ddof=0))


def test_affine_invariance_with_frozen_stats():
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

    a, b = 3.0, -1.7
    transformed = compute_ensemble_reward(
        a * scores,
        beta_policy=0.5,
        r_center=a * center,
        r_scale=a * norm_scale,
    )


    np.testing.assert_allclose(
        original.std_scores, transformed.std_scores
    )
    np.testing.assert_allclose(
        original.pessimistic_reward,
        transformed.pessimistic_reward,
    )


def test_mean_baseline_uses_same_standardization():
    scores = np.array([
        [2.0, 4.0, 3.0, 7.0, 5.0],
        [5.0, 7.0, 6.0, 8.0, 9.0],
    ])
    center = 4.0
    norm_scale = 2.5
    standardized = (scores - center) / norm_scale

    actual = compute_mean_reward(
        scores,
        r_center=center,
        r_scale=norm_scale,
    )

    np.testing.assert_allclose(actual, standardized.mean(axis=-1))


def test_zero_calibration_factor_returns_mean():
    scores = np.array([2.0, 4.0, 3.0, 7.0, 5.0])

    result = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        calibration_scale=0.0,
        r_center=3.5,
        r_scale=2.0,
    )

    np.testing.assert_allclose(
        result.pessimistic_reward, result.mean_reward
    )
    np.testing.assert_allclose(result.penalty, 0.0)


def test_penalty_nonnegative_and_decreases_with_c():
    scores = np.array([
        [0.2, 0.4, 0.1, 0.7, 0.3],
        [1.0, 0.8, 1.2, 0.9, 1.1],
    ])

    small_c = compute_ensemble_reward(
        scores, beta_policy=0.5, r_center=0.0, r_scale=1.0, c=0.5
    )
    large_c = compute_ensemble_reward(
        scores, beta_policy=0.5, r_center=0.0, r_scale=1.0, c=2.0
    )

    assert np.all(small_c.penalty >= -1e-12)
    assert np.all(large_c.penalty >= -1e-12)
    assert np.all(large_c.penalty <= small_c.penalty + 1e-12)


def test_penalty_linear_in_calibration_factor():
    scores = np.array([0.2, 0.4, 0.1, 0.7, 0.3])

    one = compute_ensemble_reward(
        scores, beta_policy=0.5, r_center=0.0, r_scale=1.0,
        calibration_scale=1.0,
    )
    two = compute_ensemble_reward(
        scores, beta_policy=0.5, r_center=0.0, r_scale=1.0,
        calibration_scale=2.0,
    )

    np.testing.assert_allclose(two.penalty, 2.0 * one.penalty)


def test_result_shapes_and_raw_scores():
    scores = np.arange(15, dtype=float).reshape(3, 5)

    result = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        r_center=2.0,
        r_scale=3.0,
    )

    assert result.raw_scores.shape == (3, 5)
    assert result.std_scores.shape == (3, 5)
    assert result.mean_reward.shape == (3,)
    assert result.variance.shape == (3,)
    assert result.pessimistic_reward.shape == (3,)
    assert result.penalty.shape == (3,)


def test_loader_reads_frozen_stats(tmp_path):
    path = tmp_path / "std_stats.json"
    path.write_text(
        json.dumps({
            "r_center": 2.5,
            "r_scale": 1.25,
            "ensemble_id": "k5-noise-off",
            "noise_condition": "noise_off",
        }),
        encoding="utf-8",
    )

    center, scale, ensemble_id, noise = load_std_stats(path)

    assert center == 2.5
    assert scale == 1.25
    assert ensemble_id == "k5-noise-off"
    assert noise == "noise_off"


def test_loader_rejects_missing_provenance(tmp_path):
    path = tmp_path / "std_stats.json"
    path.write_text(
        json.dumps({"r_center": 0.0, "r_scale": 1.0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_std_stats(path)


def test_required_normalization_arguments():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    with pytest.raises(TypeError):
        compute_ensemble_reward(scores, beta_policy=0.5)

    with pytest.raises(TypeError):
        compute_mean_reward(scores)


def test_placeholder_flag_restricts_values():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        r_center=0.0,
        r_scale=1.0,
        allow_placeholder_stats=True,
    )

    with pytest.raises(ValueError, match="placeholder"):
        compute_ensemble_reward(
            scores,
            beta_policy=0.5,
            r_center=0.2,
            r_scale=1.0,
            allow_placeholder_stats=True,
        )


def test_rejects_nonfinite_scores_and_invalid_scale():
    with pytest.raises(ValueError, match="non-finite"):
        compute_ensemble_reward(
            [0.1, 0.2, np.nan, 0.4, 0.5],
            beta_policy=0.5,
            r_center=0.0,
            r_scale=1.0,
        )

    with pytest.raises(ValueError, match="r_scale"):
        compute_ensemble_reward(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            beta_policy=0.5,
            r_center=0.0,
            r_scale=0.0,
        )
