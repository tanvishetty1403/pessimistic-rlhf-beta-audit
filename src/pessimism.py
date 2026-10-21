"""
src/pessimism.py

ensemble-pessimistic effective reward (Hahami et al. form), with beta_rob = c * beta_policy
rewards are assumed STANDARDIZED on reference samples so betas are scale-free
variance form throughout (ddof=1), so lambda = 1 / (2 * c * beta_policy) holds
"""
import numpy as np


def beta_rob(c: float, beta_policy: float) -> float:
    return c * beta_policy


def r_exact(R: np.ndarray, b_rob: float) -> np.ndarray:
    """Exact pessimistic reward: -b * log mean_k exp(-R_k / b).

    R: (..., K) ensemble rewards. Returns (...,). Numerically stable via logsumexp.
    """
    z = -R / b_rob
    m = z.max(axis=-1, keepdims=True)
    lme = np.log(np.exp(z - m).mean(axis=-1)) + m[..., 0]
    return -b_rob * lme


def r_gauss(R: np.ndarray, b_rob: float, scale: float = 1.0) -> np.ndarray:
    """Gaussian closed form: mu - scale * sigma^2 / (2 b). `scale` is the F1 calibration factor (1 = uncalibrated)."""
    mu = R.mean(axis=-1)
    var = R.var(axis=-1, ddof=1)
    return mu - scale * var / (2.0 * b_rob)


def r_mean(R: np.ndarray) -> np.ndarray:
    return R.mean(axis=-1)


def r_wco(R: np.ndarray) -> np.ndarray:
    return R.min(axis=-1)


def r_uwo(R: np.ndarray, lam: float) -> np.ndarray:
    """UWO baseline with a free, constant lambda (Coste-style): mu - lam * var."""
    return R.mean(axis=-1) - lam * R.var(axis=-1, ddof=1)


def lam_tied(c: float, beta_policy: float) -> float:
    """Effective lambda implied by the tie: 1 / (2 c beta_policy). c=1 is Hahami's default."""
    return 1.0 / (2.0 * c * beta_policy)
