"""Streaming-protocol causality guard: perturbing
the LAST label y[-1] must leave every prediction bit-identical — the
prediction at t is emitted before (x_t, y_t) enters any state."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

rls = pytest.importorskip("scripts.rls_baselines")
spar_test = pytest.importorskip("scripts.spar_test")


def _stream(seed=0, D=8, T=200, T_warm=50, sigma=0.1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, D))
    w = rng.normal(scale=0.5, size=D)
    y = x @ w + sigma * rng.normal(size=T)
    return x, y, T_warm


def _preds(out):
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def _assert_last_label_inert(runner, y_bump=1.0, **kwargs):
    x, y, T_warm = _stream()
    preds1 = _preds(runner(x, y, T_warm, **kwargs))
    y2 = y.copy()
    y2[-1] += y_bump
    preds2 = _preds(runner(x, y2, T_warm, **kwargs))
    assert preds1.shape == preds2.shape
    np.testing.assert_array_equal(preds1, preds2)


def test_run_rls_last_label_inert():
    _assert_last_label_inert(rls.run_rls, lam=0.99)


def test_run_df_last_label_inert():
    _assert_last_label_inert(rls.run_df, lam_f=0.98)


def test_run_kf_last_label_inert():
    _assert_last_label_inert(rls.run_kf, q=1e-4)


def test_run_kf_est_last_label_inert():
    _assert_last_label_inert(rls.run_kf_est)


def test_run_spar_diag_last_label_inert():
    _assert_last_label_inert(rls.run_spar_diag, refresh_every=10)


def test_run_spar_v2_last_label_inert():
    _assert_last_label_inert(spar_test.run_spar_v2, refresh_every=10, K_cal=2)


def test_run_spar_kf_last_label_inert():
    _assert_last_label_inert(
        rls.run_spar_kf, refresh_every=10, K_cal=2, return_diag=True)


def test_run_spar_kf_debias_variance_floor_last_label_inert():
    _assert_last_label_inert(
        rls.run_spar_kf, refresh_every=10, K_cal=2, debias_variance_floor=True,
        return_diag=True)


def test_classifier_stream_last_label_inert():
    from runner import classifier

    x, y, T_warm = _stream()
    y_cls = (x[:, 0] > 0).astype(int)
    K = 2
    preds1 = classifier.stream(x[:T_warm], y_cls[:T_warm], x[T_warm:],
                                y_cls[T_warm:], K, alpha=1.0, gamma=0.1,
                                reanchor_every=5)
    y2 = y_cls.copy()
    y2[-1] = (y2[-1] + 1) % K
    preds2 = classifier.stream(x[:T_warm], y_cls[:T_warm], x[T_warm:],
                                y2[T_warm:], K, alpha=1.0, gamma=0.1,
                                reanchor_every=5)
    np.testing.assert_array_equal(preds1, preds2)


def test_aal_run_stream_last_label_inert():
    import jax.numpy as jnp
    from runner import aal

    x, y, T_warm = _stream()
    cfg = aal.RunnerConfig(online=True, mode="reanchor", anchor="train",
                           reanchor_every=5)

    def run(yv):
        P0, W0, A0, B0, n0 = aal.warm_start(
            jnp.asarray(x[:T_warm]), jnp.asarray(yv[:T_warm]), cfg)
        preds, _, _, _ = aal.run_stream(
            jnp.asarray(x[T_warm:]), jnp.asarray(yv[T_warm:]),
            cfg, P0, W0, A0, B0, n0, jnp.zeros(x.shape[1]))
        return np.asarray(preds)

    preds1 = run(y)
    y2 = y.copy()
    y2[-1] += 1.0
    np.testing.assert_array_equal(preds1, run(y2))


def test_stream_spar_kf_eeg_last_label_inert():
    pytest.importorskip("pyriemann")
    pytest.importorskip("mne")
    tier1 = pytest.importorskip("scripts.bench_eeg_tier1")

    rng = np.random.default_rng(0)
    D, n_tr, n_st, K = 8, 60, 80, 3
    X_tr = rng.normal(size=(n_tr, D))
    X_st = rng.normal(size=(n_st, D))
    w = rng.normal(size=(D, K))
    y_tr = np.argmax(X_tr @ w, axis=1)
    y_st = np.argmax(X_st @ w, axis=1)

    preds1 = _preds(tier1.stream_spar_kf_eeg(X_tr, y_tr, X_st, y_st, K))
    y2 = y_st.copy()
    y2[-1] = (y2[-1] + 1) % K
    preds2 = _preds(tier1.stream_spar_kf_eeg(X_tr, y_tr, X_st, y2, K))
    np.testing.assert_array_equal(preds1, preds2)
