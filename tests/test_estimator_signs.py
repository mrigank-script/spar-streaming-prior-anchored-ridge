"""Estimator sign assertions: high measured conflict lightens
the anchor (smaller rho*); a residual CUSUM spike (drift onset) opens the
forgetting window w toward its cap."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

rls = pytest.importorskip("scripts.rls_baselines")


def _stream(seed=0, D=8, T=200, T_warm=50, drift="none"):
    return rls.make_stream_fac(seed=seed, D=D, T=T, sigma=0.1,
                               spectrum="iso", drift=drift)


def test_high_conflict_lightens_anchor():
    X, y, w_true, t1, t2, Q0, lam0 = _stream()
    T_warm = 50

    G_warm = rls.make_prior("warm", X, y, w_true, t1, T_warm, Q0, lam0)
    _, rhos_warm, _ = rls.run_spar_kf(X, y, T_warm, G_init=G_warm,
                                      refresh_every=10, K_cal=2,
                                      return_diag=True)
    G_adv = rls.make_prior("adversarial", X, y, w_true, t1, T_warm, Q0, lam0)
    _, rhos_adv, _ = rls.run_spar_kf(X, y, T_warm, G_init=G_adv,
                                    refresh_every=10, K_cal=2,
                                    return_diag=True)
    assert rhos_adv.mean() < rhos_warm.mean()
    assert rhos_adv.mean() < 0.1 * rhos_warm.mean()


def test_cusum_spike_raises_window_toward_cap():
    X, y, w_true, t1, t2, Q0, lam0 = _stream(drift="abrupt")
    T_warm = 50
    w_max = 0.05
    _, rhos, ws = rls.run_spar_kf(X, y, T_warm, refresh_every=10, K_cal=2,
                                  w_max=w_max, return_diag=True)
    drift_i = t1 - T_warm  # stream index of the abrupt drift
    pre, post = ws[:drift_i], ws[drift_i + 5:]
    assert post.mean() > pre.mean()
    assert post.max() >= 0.9 * w_max
