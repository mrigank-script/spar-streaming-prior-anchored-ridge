"""Baseline equivalence assertions: the identities that
rls_baselines._sanity() only prints, as hard asserts (max |dev| < 1e-10)."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

rls = pytest.importorskip("scripts.rls_baselines")

TOL = 1e-10


def _stream(seed=0, D=8, T=200, T_warm=50):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, D))
    w = rng.normal(scale=0.5, size=D)
    y = x @ w + 0.05 * rng.normal(size=T)
    return x, y, T_warm


def test_df_lam1_equals_rls_lam1():
    x, y, T_warm = _stream()
    p_rls = rls.run_rls(x, y, T_warm, lam=1.0, rho=0.1)
    p_df = rls.run_df(x, y, T_warm, lam_f=1.0, rho=0.1)
    assert float(np.max(np.abs(p_rls - p_df))) < TOL


def test_kf_q0_r1_equals_rls_lam1():
    x, y, T_warm = _stream()
    p_kf = rls.run_kf(x, y, T_warm, q=0.0, R=1.0, rho_init=1.0)
    p_rls = rls.run_rls(x, y, T_warm, lam=1.0, rho=1.0)
    assert float(np.max(np.abs(p_kf - p_rls))) < TOL


def test_rls_lam1_equals_batch_warm_ridge():
    x, y, T_warm = _stream()
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm]
    b = x[:T_warm].T @ y[:T_warm]
    p_rls = rls.run_rls(x, y, T_warm, lam=1.0, rho=0.1)
    batch = []
    for t in range(T_warm, len(y)):
        batch.append(x[t] @ np.linalg.solve(
            A + 0.1 * T_warm * np.eye(D), b))
        A = A + np.outer(x[t], x[t])
        b = b + x[t] * y[t]
    assert float(np.max(np.abs(p_rls - np.asarray(batch)))) < TOL


def test_wiseft_endpoints():
    x, y, T_warm = _stream()
    W_zero, W_ft = rls.wiseft_models(x, y, T_warm)
    a0 = rls.wiseft_preds(x[T_warm:], W_zero, W_ft, 0.0)
    a1 = rls.wiseft_preds(x[T_warm:], W_zero, W_ft, 1.0)
    assert float(np.max(np.abs(a0 - x[T_warm:] @ W_zero))) < TOL
    assert float(np.max(np.abs(a1 - x[T_warm:] @ W_ft))) < TOL
