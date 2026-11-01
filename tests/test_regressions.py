"""Regression tests for streaming-protocol failure modes (run: py -m pytest tests -q)."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from core.metrics import (
    cohen_kappa,
    expected_calibration_error,
    max_drawdown,
    signature_preservation_ratio,
)


def test_ece_2d_labels_match_1d():
    probs = np.array([[0.8, 0.2], [0.6, 0.4]])
    y = np.array([0, 1])
    assert expected_calibration_error(probs, y) == pytest.approx(
        expected_calibration_error(probs, y[:, None])
    )


def test_kappa_accepts_column_vectors():
    y = np.array([0, 0, 1, 1])
    assert cohen_kappa(y[:, None], y[:, None]) == pytest.approx(1.0)


def test_max_drawdown_includes_initial_capital():
    assert max_drawdown(np.array([-0.5])) == pytest.approx(-0.5)


def test_signature_metric_counts():
    assert signature_preservation_ratio(np.diag([1.0, -1.0]), np.diag([1.1, -0.9])) == 1.0
    assert signature_preservation_ratio(np.diag([1.0, -1.0]), np.diag([1.1, 0.9])) == 0.0


def test_ts_headerless_parser_restores_all_columns():
    from dataio import ts

    exchange = ts.load_raw("Exchange")
    assert exchange["values"].shape == (7588, 8)
    traffic = ts.load_raw("Traffic")
    assert traffic["values"].shape == (17544, 862)
    electricity = ts.load_raw("Electricity")
    assert electricity["values"].shape == (26304, 321)


def test_classifier_retains_warm_start():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    from runner import classifier

    W = classifier.fit_warm(X, y, 2, ridge=0.01)
    preds = classifier.stream(X, y, X, y, 2, alpha=1.0, gamma=0.05, reanchor_every=5)
    assert (preds == y).mean() > 0.95


def test_aal_ridge_scaling_invariance():
    import jax.numpy as jnp

    from runner import aal

    rng = np.random.default_rng(1)
    phi = jnp.asarray(rng.normal(size=(60, 5)))
    y = jnp.asarray(rng.normal(size=(60,)))
    cfg = aal.RunnerConfig(online=False)
    _, w_single, _, _, _ = aal.warm_start(phi, y, cfg)
    _, w_double, _, _, _ = aal.warm_start(
        jnp.concatenate([phi, phi]), jnp.concatenate([y, y]), cfg
    )
    assert np.allclose(np.asarray(w_single), np.asarray(w_double), atol=1e-8)
