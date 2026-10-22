"""
tests/test_core.py
"""

import numpy as np
from src import pessimism as P
from src import kl_ess as K

rng = np.random.default_rng(0)


def test_limits_mean_and_wco():
    R = rng.normal(size=(50, 5))
    assert np.allclose(P.r_exact(R, 1e6), P.r_mean(R), atol=1e-4)      # beta_rob -> inf = Mean
    assert np.allclose(P.r_exact(R, 1e-3), P.r_wco(R), atol=5e-3)      # beta_rob -> 0 = WCO (+ small log K term)


def test_gaussian_matches_exact_for_gaussian_ensemble():
    # large K so sample var ~ true var; moderate beta so higher cumulants negligible
    R = rng.normal(0.3, 0.4, size=(20, 200000))
    b = 2.0
    assert np.allclose(P.r_exact(R, b), P.r_gauss(R, b), atol=5e-3)


def test_tied_lambda():
    assert np.isclose(P.lam_tied(1.0, 0.5), 1.0)
    R = rng.normal(size=(10, 5))
    assert np.allclose(P.r_gauss(R, P.beta_rob(1.0, 0.5)), P.r_uwo(R, P.lam_tied(1.0, 0.5)))


def test_kl_estimator_vs_analytic_tilt():
    # r ~ N(0,1) under ref; tilt exp(r/b) -> N(1/b,1); KL = 1/(2 b^2)
    N = 400000
    r = rng.normal(size=N)
    for b in (1.0, 1.5, 2.0):
        w = K.tilt_weights(r, b)
        assert abs(K.kl_from_weights(w) - 1 / (2 * b ** 2)) < 0.02


def test_kl_cap_and_ess():
    N = 64
    r = rng.normal(size=N)
    w = K.tilt_weights(r, 1e-6)                      # ~argmax
    assert K.kl_from_weights(w) <= np.log(N) + 1e-9
    assert np.isclose(K.kl_from_weights(w), np.log(N), atol=1e-3)
    assert K.ess(w) < 1.01
    w0 = K.tilt_weights(r, 1e6)                      # ~uniform
    assert np.isclose(K.ess(w0), N, rtol=1e-3)
    assert abs(K.kl_from_weights(w0)) < 1e-6


def test_bon_kl_analytic():
    assert np.isclose(K.bon_kl_analytic(1), 0.0)
    assert np.isclose(K.bon_kl_analytic(64), np.log(64) - 63 / 64)


def test_bon_weights_match_analytic_kl():
    r = rng.normal(size=4096)
    for n in (2, 8, 64):
        w = K.bon_weights(r, n)
        assert abs(w.sum() - 1) < 1e-9
        assert abs(K.kl_from_weights(w) - K.bon_kl_analytic(n)) < 0.05
