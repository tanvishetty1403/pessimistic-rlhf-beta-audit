"""
tests/test_kl_ess_extra.py

What the BoN check is
- The KL formula  KL(pi||ref) ~= sum_i w_i log(N w_i)  treats the reference as uniform over the N cached samples.
- Best-of-n has a closed-form KL, log n - (n-1)/n, independent of the reward model. 
- Feeding the EXACT BoN selection probabilities (bon_weights) into the formula must reproduce it. 
- That validates the KL formula and its uniform-reference assumption. 
- bon_weights depends only on ranks, so the reward values play no role.
- It does NOT validate the exp(r/beta) weights (test_core.test_kl_estimator_vs_analytic_tilt does, against 1/(2 b^2)), and it says nothing about accuracy at high KL with small ESS (that is what G3 reports).
- Finite-pool effect: the empirical KL sits above log n - (n-1)/n and the gap grows with n/N, so N >> n is required.

"""
import itertools
import numpy as np
import pytest
from src import kl_ess as K


def bon_dev(N, n):
    r = np.arange(N, dtype=float)
    return K.kl_from_weights(K.bon_weights(r, n)) - K.bon_kl_analytic(n)


# ---- 1. finite-pool bias, pinned at the settings G1 actually uses
def test_bon_kl_bias_is_positive_and_grows_with_n():
    devs = [bon_dev(1024, n) for n in (2, 8, 64, 256)]
    assert all(d > -1e-9 for d in devs)
    assert devs == sorted(devs)


def test_bon_kl_bias_at_g1_range_is_below_default_tolerance():
    assert bon_dev(1024, 64) < 0.05
    assert bon_dev(1024, 256) < 0.15


def test_bon_kl_bias_shrinks_with_pool_size():
    assert bon_dev(4096, 64) < bon_dev(1024, 64) < bon_dev(256, 64)


def test_bon_weights_depend_on_ranks_only():
    r = np.random.default_rng(3).normal(size=500)
    assert np.allclose(K.bon_weights(r, 16), K.bon_weights(np.exp(3 * r) + 7, 16))


@pytest.mark.parametrize("b", [1.0, 1.5, 2.0, 3.0])
def test_ess_matches_gaussian_analytic(b):
    r = np.random.default_rng(0).normal(size=400000)
    assert abs(K.ess(K.tilt_weights(r, b)) / r.size - np.exp(-1 / b ** 2)) < 0.01


def test_batch_shapes():
    R = np.random.default_rng(1).normal(size=(3, 50))
    w = K.tilt_weights(R, 0.7)
    assert w.shape == (3, 50) and K.ess(w).shape == (3,) and K.kl_from_weights(w).shape == (3,)


def brute_bon(r, n):
    """Ground truth: average over all n-subsets; the max is taken, exact ties split uniformly."""
    N = len(r); w = np.zeros(N); subsets = list(itertools.combinations(range(N), n))
    for s in subsets:
        m = max(r[j] for j in s); win = [j for j in s if r[j] == m]
        for j in win:
            w[j] += 1.0 / len(win)
    return w / len(subsets)


def test_bon_weights_match_brute_force_without_ties():
    r = np.random.default_rng(2).normal(size=8)
    for n in (1, 3, 8):
        assert np.allclose(K.bon_weights(r, n), brute_bon(r, n))


def test_bon_weights_match_brute_force_with_ties():
    r = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0, -1.0])
    for n in (1, 2, 3, 5):
        w = K.bon_weights(r, n)
        assert np.allclose(w, brute_bon(r, n)) and abs(w.sum() - 1) < 1e-12 and (w >= 0).all()


def test_all_tied_is_uniform_with_zero_kl():
    w = K.bon_weights(np.zeros(100), 8) 
    assert np.allclose(w, 1 / 100) and abs(K.kl_from_weights(w)) < 1e-12


def test_many_tied_empties_do_not_break_weights():
    r = np.r_[np.zeros(300), np.random.default_rng(4).normal(size=724)] 
    w = K.bon_weights(r, 16)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all() and np.allclose(w[:300], w[0])


def test_bon_weights_argument_checks():
    r = np.random.default_rng(5).normal(size=10)
    with pytest.raises(ValueError): K.bon_weights(r, 0)
    with pytest.raises(ValueError): K.bon_weights(r, 11)
    with pytest.raises(ValueError): K.bon_weights(r.reshape(2, 5), 2)
    with pytest.raises(ValueError): K.bon_weights(np.r_[r, np.nan], 2)
    assert np.allclose(K.bon_weights(r, 1), 1 / 10)