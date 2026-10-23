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


def bon_weights(r: np.ndarray, n: int) -> np.ndarray:
    """Exact best-of-n selection probabilities over the N cached samples of ONE prompt. r: (N,), n: int.

    No ties: the sample with ascending rank i (1-indexed) is the max of a random n-subset w.p. C(i-1, n-1) / C(N, n).
    Ties: broken uniformly at random, which equals averaging those probabilities over the ranks a tie group
    occupies (so identical scores, e.g. repeated empty responses, share weight equally).
    Raises ValueError unless 1 <= n <= N, r is 1-D, and r is finite. Returns weights aligned with r, sum to 1.
    Feeding them to kl_from_weights checks the KL formula against log n - (n-1)/n (finite-pool bias grows with n/N).
    """
    from scipy.special import gammaln
    r = np.asarray(r)
    if r.ndim != 1:
        raise ValueError("bon_weights takes one prompt's scores, shape (N,); loop over prompts")
    N = r.shape[0]
    if not (1 <= n <= N):
        raise ValueError(f"need 1 <= n <= N, got n={n}, N={N}")
    if not np.isfinite(r).all():
        raise ValueError("non-finite scores")
    order = np.argsort(r, kind="stable")
    rs = r[order]
    i = np.arange(1, N + 1)
    with np.errstate(invalid="ignore"):
        logc = (gammaln(i) - gammaln(n) - gammaln(i - n + 1)) - (gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1))
    p_sorted = np.where(i >= n, np.exp(np.nan_to_num(logc, nan=-np.inf)), 0.0)
    starts = np.r_[0, np.flatnonzero(rs[1:] != rs[:-1]) + 1]            # first index of each tie group
    counts = np.diff(np.r_[starts, N])
    p_sorted = np.repeat(np.add.reduceat(p_sorted, starts) / counts, counts)   # no-op when there are no ties
    w = np.zeros(N)
    w[order] = p_sorted
    return w