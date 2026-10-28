
"""K=5 ensemble reward interface for the RLOO feasibility experiment.

Normalization:
- Fit statistics on reference samples only, using compute_std_stats().
- Freeze and load the statistics for subsequent online reward calls.
- Use the same statistics for every member of an ensemble.
- Do not substitute online-batch statistics for reference statistics.

Calibration:
- calibration_scale is the G2/F1 variance-penalty factor.
- Fit it using rewards in the same standardized units.

Failure policy:
- Non-finite reward-model scores raise ValueError.
- Scores are not silently clamped; this is deliberate fail-loud behavior.
- If fp16 overflow occurs, investigate its source before changing policy.

Placeholder policy:
- (r_center=0, r_scale=1) is allowed only when explicitly opted into
  with allow_placeholder_stats=True. It is never the default.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src import pessimism as P


EXPECTED_MEMBERS = 5


@dataclass(frozen=True)
class EnsembleRewardResult:
    """Auditable rewards and metadata for one batch."""

    raw_scores: NDArray[np.float64]
    std_scores: NDArray[np.float64]
    mean_reward: NDArray[np.float64]
    variance: NDArray[np.float64]
    pessimistic_reward: NDArray[np.float64]
    penalty: NDArray[np.float64]
    c: float
    calibration_scale: float
    r_center: float
    r_scale: float
    ensemble_id: str
    noise_condition: str


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
        raise ValueError(
            "member_scores contain non-finite values; "
            "failing loudly rather than clamping"
        )

    return scores


def compute_std_stats(R_ref: ArrayLike) -> dict[str, float]:
    """Fit global center/scale from reference ensemble scores.

    R_ref has shape (..., K). Compute the ensemble mean over K first,
    then fit one global center and population standard deviation over
    all reference samples. This mirrors the specified global convention
    from the offline study.

    Call this only on the designated reference data, never on online
    training batches. Save the returned statistics for later reuse.
    """
    scores = np.asarray(R_ref, dtype=np.float64)

    if scores.ndim < 2 or scores.shape[-1] < 2:
        raise ValueError("R_ref must have shape (..., K), with K >= 2")

    if not np.all(np.isfinite(scores)):
        raise ValueError("R_ref must contain only finite values")

    ensemble_means = scores.mean(axis=-1).reshape(-1)
    center = float(ensemble_means.mean())
    scale = float(ensemble_means.std(ddof=0))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Reference ensemble-mean standard deviation must be positive")

    return {"r_center": center, "r_scale": scale}


def load_std_stats(
    path: str | Path,
) -> tuple[float, float, str, str]:
    """Load frozen statistics and their ensemble/noise provenance.

    Expected JSON keys:
      r_center, r_scale, ensemble_id, noise_condition
    """
    with Path(path).open("r", encoding="utf-8") as file:
        stats = json.load(file)

    required = {"r_center", "r_scale", "ensemble_id", "noise_condition"}
    missing = required - stats.keys()
    if missing:
        raise ValueError(
            f"Statistics file is missing required fields: {sorted(missing)}"
        )

    center = float(stats["r_center"])
    scale = float(stats["r_scale"])
    ensemble_id = str(stats["ensemble_id"])
    noise_condition = str(stats["noise_condition"])

    if not np.isfinite(center):
        raise ValueError("r_center must be finite")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("r_scale must be finite and positive")
    if not ensemble_id.strip() or not noise_condition.strip():
        raise ValueError("ensemble_id and noise_condition must be non-empty")

    return center, scale, ensemble_id, noise_condition


def compute_ensemble_reward(
    member_scores: ArrayLike,
    *,
    beta_policy: float,
    r_center: float,
    r_scale: float,
    c: float = 1.0,
    calibration_scale: float = 1.0,
    expected_members: int = EXPECTED_MEMBERS,
    allow_placeholder_stats: bool = False,
    ensemble_id: str = "unspecified",
    noise_condition: str = "unspecified",
) -> EnsembleRewardResult:
    """Standardize scores and compute Gaussian pessimistic rewards.

    r_center and r_scale are required keyword arguments. Explicitly
    enabling allow_placeholder_stats permits (0, 1) for smoke tests
    only; real runs must load frozen statistics from the config's file.
    """
    raw_scores = _validate_scores(member_scores, expected_members)

    if not np.isfinite(beta_policy) or beta_policy <= 0:
        raise ValueError("beta_policy must be finite and positive")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")
    if not np.isfinite(r_center):
        raise ValueError("r_center must be finite")
    if not np.isfinite(r_scale) or r_scale <= 0:
        raise ValueError("r_scale must be finite and positive")
    if not np.isfinite(calibration_scale) or calibration_scale < 0:
        raise ValueError(
            "calibration_scale must be finite and non-negative"
        )

    if allow_placeholder_stats and (r_center != 0.0 or r_scale != 1.0):
        raise ValueError(
            "allow_placeholder_stats requires r_center=0 and r_scale=1"
        )

    std_scores = (raw_scores - r_center) / r_scale
    mean_reward = P.r_mean(std_scores)
    variance = std_scores.var(axis=-1, ddof=1)
    pessimistic_reward = P.r_gauss(
        std_scores,
        P.beta_rob(c, beta_policy),
        scale=calibration_scale,
    )

    return EnsembleRewardResult(
        raw_scores=raw_scores,
        std_scores=std_scores,
        mean_reward=mean_reward,
        variance=variance,
        pessimistic_reward=pessimistic_reward,
        penalty=mean_reward - pessimistic_reward,
        c=c,
        calibration_scale=calibration_scale,
        r_center=float(r_center),
        r_scale=float(r_scale),
        ensemble_id=ensemble_id,
        noise_condition=noise_condition,
    )


def compute_mean_reward(
    member_scores: ArrayLike,
    *,
    r_center: float,
    r_scale: float,
    expected_members: int = EXPECTED_MEMBERS,
    allow_placeholder_stats: bool = False,
) -> NDArray[np.float64]:
    """Return the mean baseline using the same frozen normalization."""
    scores = _validate_scores(member_scores, expected_members)

    if not np.isfinite(r_center):
        raise ValueError("r_center must be finite")
    if not np.isfinite(r_scale) or r_scale <= 0:
        raise ValueError("r_scale must be finite and positive")
    if allow_placeholder_stats and (r_center != 0.0 or r_scale != 1.0):
        raise ValueError(
            "allow_placeholder_stats requires r_center=0 and r_scale=1"
        )

    std_scores = (scores - r_center) / r_scale
    return P.r_mean(std_scores)
