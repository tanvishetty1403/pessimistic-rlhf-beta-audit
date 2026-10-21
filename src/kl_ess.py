"""
src/kl_ess.py

Weighted-resampling policy approximation: KL and ESS from self-normalized weights
For one prompt with N cached reference samples, the approximate KL-optimal policy puts weight w_i directly-proportional-to exp(r~_i / beta_policy) on sample i
Reference is uniform over the N samples, so KL(pi || ref) ≈ sum_i w_i * log(N * w_i)
Reachable KL is capped at log N
"""
import numpy as np


def tilt_weights(r_tilde: np.ndarray, beta_policy: float) -> np.ndarray:
    """r_tilde: (..., N). Returns normalized weights (..., N)."""
    z = r_tilde / beta_policy
    z = z - z.max(axis=-1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=-1, keepdims=True)


def kl_from_weights(w: np.ndarray) -> np.ndarray:
    n = w.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(w > 0, w * np.log(n * w), 0.0)
    return t.sum(axis=-1)


def ess(w: np.ndarray) -> np.ndarray:
    return 1.0 / (w ** 2).sum(axis=-1)


def bon_kl_analytic(n: int) -> float:
    """Analytic KL of best-of-n vs reference: log n - (n-1)/n."""
    return float(np.log(n) - (n - 1) / n)


def expected_gold(w: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Expected gold reward under the weighted policy, per prompt. w, gold: (..., N)."""
    return (w * gold).sum(axis=-1)
