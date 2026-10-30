
import json

import numpy as np
import pytest

from src import pessimism as P
from src.rloo_reward import (
    StdStats,
    compute_ensemble_reward,
    compute_mean_reward,
    compute_std_stats,
    load_std_stats,
    save_std_stats,
)


def make_stats(
    center=0.25,
    scale=0.8,
    ensemble_id="test-k5",
    noise_condition="noise_off",
):
    return StdStats(
        r_center=center,
        r_scale=scale,
        ensemble_id=ensemble_id,
        noise_condition=noise_condition,
    )


def test_reward_matches_manually_standardized_scores():
    raw = np.array([2.0, 4.0, 3.0, 7.0, 5.0])
    stats = make_stats(center=3.5, scale=2.0)
    beta_policy = 0.7
    c = 1.3
    calibration_scale = 0.8

    std = (raw - stats.r_center) / stats.r_scale

    result = compute_ensemble_reward(
        raw,
        beta_policy=beta_policy,
        stats=stats,
        c=c,
        calibration_scale=calibration_scale,
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

    assert result.beta_policy == beta_policy
    assert result.c == c
    assert result.calibration_scale == calibration_scale
    assert result.r_center == stats.r_center
    assert result.r_scale == stats.r_scale
    assert result.ensemble_id == stats.ensemble_id
    assert result.noise_condition == stats.noise_condition
    assert result.placeholder_stats is False


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

    stats_dict = compute_std_stats(reference)
    ensemble_means = reference.mean(axis=-1)
    standardized_means = (
        ensemble_means - stats_dict["r_center"]
    ) / stats_dict["r_scale"]

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
    stats_dict = compute_std_stats(reference)

    np.testing.assert_allclose(
        stats_dict["r_center"], means.mean()
    )
    np.testing.assert_allclose(
        stats_dict["r_scale"], means.std(ddof=0)
    )


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
        stats=make_stats(center, norm_scale),
    )

    a = 3.0
    b = -1.7

    transformed = compute_ensemble_reward(
        a * scores + b,
        beta_policy=0.5,
        stats=make_stats(
            center=a * center + b,
            scale=a * norm_scale,
        ),
    )

    np.testing.assert_allclose(
        original.std_scores, transformed.std_scores
    )
    np.testing.assert_allclose(
        original.mean_reward, transformed.mean_reward
    )
    np.testing.assert_allclose(
        original.variance, transformed.variance
    )
    np.testing.assert_allclose(
        original.pessimistic_reward,
        transformed.pessimistic_reward,
    )
    np.testing.assert_allclose(
        original.penalty, transformed.penalty
    )


def test_mean_baseline_uses_same_standardization():
    scores = np.array([
        [2.0, 4.0, 3.0, 7.0, 5.0],
        [5.0, 7.0, 6.0, 8.0, 9.0],
    ])

    stats = make_stats(center=4.0, scale=2.5)
    standardized = (scores - stats.r_center) / stats.r_scale

    actual = compute_mean_reward(scores, stats=stats)

    np.testing.assert_allclose(
        actual, standardized.mean(axis=-1)
    )


def test_zero_calibration_factor_returns_mean():
    scores = np.array([2.0, 4.0, 3.0, 7.0, 5.0])

    result = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        calibration_scale=0.0,
        stats=make_stats(center=3.5, scale=2.0),
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
    stats = make_stats()

    small_c = compute_ensemble_reward(
        scores, beta_policy=0.5, stats=stats, c=0.5
    )
    large_c = compute_ensemble_reward(
        scores, beta_policy=0.5, stats=stats, c=2.0
    )

    assert np.all(small_c.penalty >= -1e-12)
    assert np.all(large_c.penalty >= -1e-12)
    assert np.all(
        large_c.penalty <= small_c.penalty + 1e-12
    )


def test_penalty_linear_in_calibration_factor():
    scores = np.array([0.2, 0.4, 0.1, 0.7, 0.3])
    stats = make_stats()

    one = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        stats=stats,
        calibration_scale=1.0,
    )
    two = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        stats=stats,
        calibration_scale=2.0,
    )

    np.testing.assert_allclose(
        two.penalty, 2.0 * one.penalty
    )


def test_result_shapes_and_raw_scores():
    scores = np.arange(15, dtype=float).reshape(3, 5)

    result = compute_ensemble_reward(
        scores,
        beta_policy=0.5,
        stats=make_stats(),
    )

    assert result.raw_scores.shape == (3, 5)
    assert result.std_scores.shape == (3, 5)
    assert result.mean_reward.shape == (3,)
    assert result.variance.shape == (3,)
    assert result.pessimistic_reward.shape == (3,)
    assert result.penalty.shape == (3,)


def test_save_and_load_std_stats(tmp_path):
    path = tmp_path / "std_stats.json"
    stats_dict = {"r_center": 2.5, "r_scale": 1.25}

    saved = save_std_stats(
        path,
        stats_dict,
        ensemble_id="k5-noise-off",
        noise_condition="noise_off",
        n_ref=200,
    )

    loaded = load_std_stats(path)

    assert saved == loaded
    assert loaded.r_center == 2.5
    assert loaded.r_scale == 1.25
    assert loaded.ensemble_id == "k5-noise-off"
    assert loaded.noise_condition == "noise_off"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_ref"] == 200


def test_save_std_stats_rejects_invalid_noise_condition(tmp_path):
    with pytest.raises(ValueError, match="noise_condition"):
        save_std_stats(
            tmp_path / "bad.json",
            {"r_center": 0.5, "r_scale": 1.0},
            ensemble_id="k5",
            noise_condition="unknown",
            n_ref=200,
        )


def test_loader_rejects_missing_provenance(tmp_path):
    path = tmp_path / "std_stats.json"
    path.write_text(
        json.dumps({
            "r_center": 0.0,
            "r_scale": 1.0,
            "ensemble_id": "k5",
            "noise_condition": "noise_off",
            "n_ref": 200,
        }),
        encoding="utf-8",
    )

    # The file has complete fields, but these are placeholder values.
    # Loading is allowed; using them in a reward call is what is guarded.
    loaded = load_std_stats(path)
    assert loaded.r_center == 0.0
    assert loaded.r_scale == 1.0


def test_loader_rejects_bad_scale(tmp_path):
    path = tmp_path / "bad_stats.json"
    path.write_text(
        json.dumps({
            "r_center": 0.5,
            "r_scale": 0.0,
            "ensemble_id": "k5",
            "noise_condition": "noise_off",
            "n_ref": 200,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="r_scale"):
        load_std_stats(path)


def test_loader_rejects_missing_provenance(tmp_path):
    path = tmp_path / "std_stats.json"
    path.write_text(
        json.dumps({
            "r_center": 0.5,
            "r_scale": 1.0,
            "n_ref": 200,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_std_stats(path)


def test_loader_allows_placeholder_stats(tmp_path):
    path = tmp_path / "placeholder_stats.json"
    path.write_text(
        json.dumps({
            "r_center": 0.0,
            "r_scale": 1.0,
            "ensemble_id": "k5",
            "noise_condition": "noise_off",
            "n_ref": 200,
        }),
        encoding="utf-8",
    )

    loaded = load_std_stats(path)

    assert loaded.r_center == 0.0
    assert loaded.r_scale == 1.0
    assert loaded.ensemble_id == "k5"
    assert loaded.noise_condition == "noise_off"


def test_rejects_wrong_ensemble_size():
    with pytest.raises(ValueError, match="Expected 5"):
        compute_ensemble_reward(
            [0.1, 0.2, 0.3],
            beta_policy=0.5,
            stats=make_stats(),
        )


def test_rejects_invalid_beta():
    with pytest.raises(ValueError, match="beta_policy"):
        compute_ensemble_reward(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            beta_policy=0.0,
            stats=make_stats(),
        )


def test_rejects_invalid_c():
    with pytest.raises(ValueError, match="c must be finite and positive"):
        compute_ensemble_reward(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            beta_policy=0.5,
            c=0.0,
            stats=make_stats(),
        )


def test_compute_std_stats_rejects_k_less_than_two():
    with pytest.raises(ValueError, match="at least two"):
        compute_std_stats(np.ones((10, 1)))


def test_compute_std_stats_rejects_non_finite_values():
    reference = np.ones((10, 5))
    reference[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_std_stats(reference)


def test_compute_std_stats_rejects_zero_scale():
    reference = np.ones((10, 5))

    with pytest.raises(ValueError, match="standard deviation"):
        compute_std_stats(reference)


def test_placeholder_stats_require_explicit_permission():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    placeholder = make_stats(
        center=0.0,
        scale=1.0,
        ensemble_id="smoke-test-placeholder",
        noise_condition="noise_off",
    )

    with pytest.raises(ValueError, match="Placeholder statistics"):
        compute_ensemble_reward(
            member_scores=scores,
            beta_policy=0.5,
            stats=placeholder,
        )

    result = compute_ensemble_reward(
        member_scores=scores,
        beta_policy=0.5,
        stats=placeholder,
        allow_placeholder_stats=True,
    )

    assert result.placeholder_stats is True


def test_placeholder_flag_rejects_non_placeholder_stats():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    with pytest.raises(ValueError, match="only valid"):
        compute_ensemble_reward(
            member_scores=scores,
            beta_policy=0.5,
            stats=make_stats(),
            allow_placeholder_stats=True,
        )


def test_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="non-finite"):
        compute_ensemble_reward(
            [0.1, 0.2, np.nan, 0.4, 0.5],
            beta_policy=0.5,
            stats=make_stats(),
        )


def test_rejects_invalid_stats_type():
    with pytest.raises(TypeError, match="StdStats"):
        compute_ensemble_reward(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            beta_policy=0.5,
            stats={"r_center": 0.5, "r_scale": 1.0},
        )


def test_mean_baseline_rejects_placeholder_without_permission():
    with pytest.raises(ValueError, match="Placeholder statistics"):
        compute_mean_reward(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            stats=make_stats(center=0.0, scale=1.0),
        )
