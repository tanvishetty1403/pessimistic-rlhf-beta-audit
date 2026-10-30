
"""K=5 ensemble reward interface for the RLOO feasibility experiment.

Normalization:
- Fit statistics on reference samples using compute_std_stats().
- Freeze and load statistics for subsequent online reward calls.
- The normalization statistics are tied to ensemble identity and noise condition.
- Never fit normalization statistics on online batches.

Calibration:
- calibration_scale is the G2/F1 variance-penalty calibration factor.
- Fit it using rewards in the same standardized units.

Failure policy:
- Non-finite reward-model scores raise ValueError.
- Scores are not silently clamped.
- Run the reward model in fp32 to reduce the risk of fp16 overflow.

Placeholder policy:
- r_center=0 and r_scale=1 are placeholders only.
- They are rejected unless allow_placeholder_stats=True.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src import pessimism as P


EXPECTED_MEMBERS = 5
VALID_NOISE_CONDITIONS = {"noise_on", "noise_off"}


@dataclass(frozen=True)
class StdStats:
    """Frozen normalization statistics with ensemble provenance."""

    r_center: float
    r_scale: float
    ensemble_id: str
    noise_condition: str

    def __post_init__(self):
        if not np.isfinite(self.r_center):
            raise ValueError("r_center must be finite")

        if not np.isfinite(self.r_scale) or self.r_scale <= 0:
            raise ValueError("r_scale must be finite and positive")

        if not self.ensemble_id.strip():
            raise ValueError("ensemble_id must be non-empty")

        if self.noise_condition not in VALID_NOISE_CONDITIONS:
            raise ValueError(
                "noise_condition must be 'noise_on' or 'noise_off'"
            )


@dataclass(frozen=True)
class EnsembleRewardResult:
    """Auditable rewards, inputs, and parameters for one batch."""

    raw_scores: NDArray[np.float64]
    std_scores: NDArray[np.float64]
    mean_reward: NDArray[np.float64]
    variance: NDArray[np.float64]
    pessimistic_reward: NDArray[np.float64]
    penalty: NDArray[np.float64]

    beta_policy: float
    c: float
    calibration_scale: float

    r_center: float
    r_scale: float
    ensemble_id: str
    noise_condition: str
    placeholder_stats: bool


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


def _validate_stats(
    stats: StdStats,
    allow_placeholder_stats: bool,
) -> bool:
    if not isinstance(stats, StdStats):
        raise TypeError("stats must be a StdStats instance")

    # Value-based detection is intentional for smoke-test placeholders.
    # A real reference distribution could theoretically produce exactly
    # (0.0, 1.0), in which case it would also be classified as a placeholder.
    is_placeholder = stats.r_center == 0.0 and stats.r_scale == 1.0

    if is_placeholder and not allow_placeholder_stats:
        raise ValueError(
            "Placeholder statistics (r_center=0, r_scale=1) "
            "require allow_placeholder_stats=True"
        )

    if allow_placeholder_stats and not is_placeholder:
        raise ValueError(
            "allow_placeholder_stats=True is only valid with "
            "r_center=0 and r_scale=1"
        )

    return is_placeholder


def compute_std_stats(R_ref: ArrayLike) -> dict[str, float | int]:
    """Fit global center/scale and count reference samples.

    R_ref has shape (..., K). Compute the ensemble mean over K first,
    then fit one global center and population standard deviation over
    all reference samples.

    Returns:
        A dictionary containing:
        - r_center: Global mean of reference ensemble means.
        - r_scale: Population standard deviation of reference ensemble means.
        - n_ref: Number of reference samples used to fit the statistics.

    This mirrors the global standardization convention in
    g1_g3_plots.py. Call this only on designated reference data.
    """
    scores = np.asarray(R_ref, dtype=np.float64)

    if scores.ndim < 2:
        raise ValueError("R_ref must have shape (..., K)")

    if scores.shape[-1] < 2:
        raise ValueError("R_ref must contain at least two ensemble members")

    if not np.all(np.isfinite(scores)):
        raise ValueError("R_ref must contain only finite values")

    ensemble_means = scores.mean(axis=-1).reshape(-1)
    center = float(ensemble_means.mean())
    scale = float(ensemble_means.std(ddof=0))
    n_ref = int(ensemble_means.size)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Reference ensemble-mean standard deviation must be positive"
        )

    return {
        "r_center": center,
        "r_scale": scale,
        "n_ref": n_ref,
    }


def save_std_stats(
    path: str | Path,
    stats_dict: dict,
    ensemble_id: str,
    noise_condition: str,
) -> StdStats:
    """Validate and save frozen statistics with their provenance."""
    if not isinstance(stats_dict, dict):
        raise TypeError("stats_dict must be a dictionary")

    required = {"r_center", "r_scale", "n_ref"}
    missing = required - stats_dict.keys()

    if missing:
        raise ValueError(
            f"stats_dict is missing required fields: {sorted(missing)}"
        )

    n_ref = stats_dict["n_ref"]

    if (
        not isinstance(n_ref, (int, np.integer))
        or isinstance(n_ref, (bool, np.bool_))
        or n_ref <= 0
    ):
        raise ValueError("n_ref must be a positive integer")

    stats = StdStats(
        r_center=float(stats_dict["r_center"]),
        r_scale=float(stats_dict["r_scale"]),
        ensemble_id=ensemble_id,
        noise_condition=noise_condition,
    )

    payload = {
        "r_center": stats.r_center,
        "r_scale": stats.r_scale,
        "ensemble_id": stats.ensemble_id,
        "noise_condition": stats.noise_condition,
        "n_ref": int(n_ref),
    }

    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    return stats


def load_std_stats(path: str | Path) -> StdStats:
    """Load and validate frozen statistics and their provenance."""
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    required = {
        "r_center",
        "r_scale",
        "ensemble_id",
        "noise_condition",
        "n_ref",
    }
    missing = required - payload.keys()

    if missing:
        raise ValueError(
            f"Statistics file is missing required fields: {sorted(missing)}"
        )

    n_ref = payload["n_ref"]
    if not isinstance(n_ref, int) or isinstance(n_ref, bool) or n_ref <= 0:
        raise ValueError("n_ref in statistics file must be a positive integer")

    return StdStats(
        r_center=float(payload["r_center"]),
        r_scale=float(payload["r_scale"]),
        ensemble_id=str(payload["ensemble_id"]),
        noise_condition=str(payload["noise_condition"]),
    )


def compute_ensemble_reward(
    member_scores: ArrayLike,
    *,
    beta_policy: float,
    stats: StdStats,
    c: float = 1.0,
    calibration_scale: float = 1.0,
    expected_members: int = EXPECTED_MEMBERS,
    allow_placeholder_stats: bool = False,
) -> EnsembleRewardResult:
    """Standardize scores and compute Gaussian pessimistic rewards.

    The normalization values and provenance must be supplied together
    through a validated StdStats object.

    Placeholder statistics are rejected unless explicitly enabled.
    """
    raw_scores = _validate_scores(member_scores, expected_members)
    placeholder_stats = _validate_stats(stats, allow_placeholder_stats)

    if not np.isfinite(beta_policy) or beta_policy <= 0:
        raise ValueError("beta_policy must be finite and positive")

    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")

    if not np.isfinite(calibration_scale) or calibration_scale < 0:
        raise ValueError(
            "calibration_scale must be finite and non-negative"
        )

    std_scores = (raw_scores - stats.r_center) / stats.r_scale
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
        beta_policy=float(beta_policy),
        c=float(c),
        calibration_scale=float(calibration_scale),
        r_center=stats.r_center,
        r_scale=stats.r_scale,
        ensemble_id=stats.ensemble_id,
        noise_condition=stats.noise_condition,
        placeholder_stats=placeholder_stats,
    )


def compute_mean_reward(
    member_scores: ArrayLike,
    *,
    stats: StdStats,
    expected_members: int = EXPECTED_MEMBERS,
    allow_placeholder_stats: bool = False,
) -> NDArray[np.float64]:
    """Return the mean baseline using the same frozen normalization.

    Use the exact same StdStats instance for the Mean baseline and
    pessimistic reward runs.
    """
    scores = _validate_scores(member_scores, expected_members)
    _validate_stats(stats, allow_placeholder_stats)

    std_scores = (scores - stats.r_center) / stats.r_scale
    return P.r_mean(std_scores)
