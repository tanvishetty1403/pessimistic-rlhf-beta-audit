
"""Reward interface for the K=5 RLOO feasibility experiment.

IMPORTANT:
- r_center=0 and r_scale=1 are smoke-test placeholders only.
- Real values must be loaded from frozen reference-sample statistics.
- Noise-on and noise-off ensembles may require different statistics.
- The F1 calibration factor is separate from normalization.
- KL handling belongs to the training loop, not this module.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src import pessimism as P


EXPECTED_MEMBERS = 5


@dataclass(frozen=True)
class EnsembleRewardResult:
    """Reward values and diagnostics for a batch."""

    member_scores: NDArray[np.float64]
    mean_reward: NDArray[np.float64]
    variance: NDArray[np.float64]
    pessimistic_reward: NDArray[np.float64]
    penalty: NDArray[np.float64]


def _validate_scores(
    member_scores: ArrayLike,
    expected_members: int,
) -> NDArray[np.float64]:
    scores = np.asarray(member_scores, dtype=np.float64)

    if scores.ndim not in (1, 2):
        raise ValueError(
            "member_scores must have shape (K,) or (batch_size, K)"
        )

    if expected_members < 2:
        raise ValueError("expected_members must be at least 2")

    if scores.shape[-1] != expected_members:
        raise ValueError(
            f"Expected {expected_members} ensemble members, "
            f"received {scores.shape[-1]}"
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("member_scores must contain only finite values")

    return scores


def compute_ensemble_reward(
    member_scores: ArrayLike,
    *,
    beta_policy: float,
    c: float = 1.0,
    r_center: float = 0.0,
    r_scale: float = 1.0,
    calibration_scale: float = 1.0,
    expected_members: int = EXPECTED_MEMBERS,
    smoke_test_placeholder: bool = False,
) -> EnsembleRewardResult:
    """Standardize ensemble scores and calculate the Gaussian reward.

    Set smoke_test_placeholder=True only for tests using r_center=0,
    r_scale=1. These values are not real calibration statistics.
    """
    scores = _validate_scores(member_scores, expected_members)

    if not np.isfinite(beta_policy) or beta_policy <= 0:
        raise ValueError("beta_policy must be finite and positive")

    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")

    if not np.isfinite(r_center):
        raise ValueError("r_center must be finite")

    if not np.isfinite(r_scale) or r_scale <= 0:
        raise ValueError("r_scale must be finite and positive")

    if (
        not np.isfinite(calibration_scale)
        or calibration_scale < 0
    ):
        raise ValueError(
            "calibration_scale must be finite and non-negative"
        )

    if smoke_test_placeholder:
        if r_center != 0.0 or r_scale != 1.0:
            raise ValueError(
                "Smoke-test placeholder mode requires r_center=0 "
                "and r_scale=1"
            )

    standardized_scores = (scores - r_center) / r_scale

    mean_reward = P.r_mean(standardized_scores)
    variance = standardized_scores.var(axis=-1, ddof=1)

    b_rob = P.beta_rob(c, beta_policy)
    pessimistic_reward = P.r_gauss(
        standardized_scores,
        b_rob,
        scale=calibration_scale,
    )

    return EnsembleRewardResult(
        member_scores=standardized_scores,
        mean_reward=mean_reward,
        variance=variance,
        pessimistic_reward=pessimistic_reward,
        penalty=mean_reward - pessimistic_reward,
    )


def compute_mean_reward(
    member_scores: ArrayLike,
    *,
    expected_members: int = EXPECTED_MEMBERS,
) -> NDArray[np.float64]:
    """Return the ensemble mean without a pessimism penalty."""
    scores = _validate_scores(member_scores, expected_members)
    return P.r_mean(scores)
