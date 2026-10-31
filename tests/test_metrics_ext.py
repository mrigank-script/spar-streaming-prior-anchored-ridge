"""Tests for the diagnostic metrics (core/metrics_ext.py)."""

import numpy as np

from core.metrics_ext import (
    anchor_reliance,
    aroa,
    backward_alignment_score,
    certified_conditioning,
    drift_kinematics,
)


def test_drift_kinematics_constant_and_ramp():
    v, a = drift_kinematics(np.ones(50))
    assert np.allclose(v, 0.0) and np.allclose(a, 0.0)
    v, a = drift_kinematics(np.arange(50, dtype=float))
    assert np.allclose(v[2:-2], 1.0) and np.allclose(a[2:-2], 0.0)


def test_aroa_signs_and_scaling():
    assert aroa([2.0] * 10, [1.0] * 10, n_actions=2) == 0.5
    assert aroa([1.0] * 10, [2.0] * 10, n_actions=1) == -1.0
    assert np.isnan(aroa([1.0], [1.0], n_actions=0))
    assert aroa([2.0] * 10, [1.0] * 10, n_actions=2, far=0.5, cost_per_action=0.2) == 0.4


def test_anchor_reliance_limits():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(10, 10))
    A = A.T @ A
    B = rng.normal(size=10)
    G = rng.normal(size=10)
    assert anchor_reliance(A, B, n=100.0, gamma=0.0, ridge=0.01, G=G) == 0.0
    assert anchor_reliance(A, B, n=100.0, gamma=1e12, ridge=0.01, G=G) > 0.999


def test_certified_conditioning_padded_tight_and_truncated_caught():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(6, 3))
    B = M[:, :2] @ rng.normal(size=(2, 4))
    res_ok = certified_conditioning(B, gamma=0.5, mode="padded")
    assert res_ok["margin"] <= 1e-6 + 1e-6 * res_ok["kappa_bound"]
    assert res_ok["sigma_min_deployed"] >= 0.5 - 1e-8
    res_bad = certified_conditioning(B, gamma=0.5, mode="rank-truncated")
    assert res_bad["margin"] < -1e3


def test_backward_alignment_score_extremes():
    rng = np.random.default_rng(2)
    blocks = [rng.normal(size=(16, 16)) for _ in range(4)]
    G_block = np.zeros((64, 64))
    for i, b in enumerate(blocks):
        G_block[i * 16:(i + 1) * 16, i * 16:(i + 1) * 16] = b
    assert backward_alignment_score(G_block, block=16) > 0.999
    G_dense = rng.normal(size=(64, 64))
    assert backward_alignment_score(G_dense, block=16) < 0.5
    off = np.ones((64, 64)) - np.kron(np.eye(4), np.ones((16, 16)))
    assert backward_alignment_score(off, block=16) < 0.01
