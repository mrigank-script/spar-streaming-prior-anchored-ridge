"""Diagnostic metrics for the drift and streaming experiments.

- drift_kinematics: velocity/acceleration of a drift-score stream; the
  acceleration is ~0 under stationarity, which gives a null distribution.
- aroa: Adaptation Return on Action -- error reduction per adaptation action
  (economics of test-time adaptation).
- anchor_reliance: exact prior/data decomposition of the anchored solve
  (streaming telemetry).
- certified_conditioning: condition-number certificate from the SV-shift lemma
  plus the deployment margin (runtime check that spectral anchoring is correct).
- backward_alignment_score: block-diagonal compressibility of gradient
  second-moments (predicts the ceiling of no-backprop local readouts).
"""

import numpy as np


def drift_kinematics(scores, dt: float = 1.0):
    """Velocity and acceleration (finite differences) of a drift score stream."""
    s = np.asarray(scores, dtype=float)
    v = np.gradient(s, dt)
    a = np.gradient(v, dt)
    return v, a


def aroa(errors_no_adapt, errors_adapt, n_actions: float, far: float = 0.0,
         cost_per_action: float = 0.0):
    """Adaptation Return on Action: (E_none - E_adapt)/N_actions - FAR*cost."""
    e0 = float(np.mean(errors_no_adapt))
    e1 = float(np.mean(errors_adapt))
    if n_actions <= 0:
        return float("nan")
    return (e0 - e1) / float(n_actions) - far * cost_per_action


def anchor_reliance(A, B, n, gamma: float, ridge: float, G):
    """ARR = ||W_prior|| / (||W_data|| + ||W_prior||), exact decomposition of
    W = (A + n(gamma+ridge)I)^{-1} (B + n*gamma*G)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    G = np.asarray(G, dtype=float)
    M = A + float(n) * (gamma + ridge) * np.eye(A.shape[0])
    W_data = np.linalg.solve(M, B)
    W_prior = np.linalg.solve(M, float(n) * gamma * G)
    denom = np.linalg.norm(W_data) + np.linalg.norm(W_prior) + 1e-12
    return float(np.linalg.norm(W_prior) / denom)


def _padded_polar(B):
    """Polar factor from the full SVD: lifts all min(D,K) singular values."""
    D, K = B.shape
    U, _, Vh = np.linalg.svd(B, full_matrices=True)
    if D >= K:
        return U[:, :K] @ Vh[:K, :]
    return U @ Vh[:D, :]


def certified_conditioning(B, gamma: float, mode: str = "padded"):
    """Kappa certificate and deployment margin for the spectral anchor.

    mode='padded' deploys the full-SVD polar anchor (all min(D,K) singular
    values lifted by gamma; margin ~ 0 means the certificate is tight).
    mode='rank-truncated' deploys only the numerical-rank directions: zero
    singular values stay zero, kappa_actual is large and the margin goes
    strongly negative -- the runtime assertion catches it.
    """
    B = np.asarray(B, dtype=float)
    sv0 = np.linalg.svd(B, compute_uv=False)
    if mode == "padded":
        Q = _padded_polar(B)
    else:
        r = int(np.sum(sv0 > 1e-10 * max(sv0[0], 1e-12)))
        U, _, Vh = np.linalg.svd(B, full_matrices=False)
        Q = U[:, :r] @ Vh[:r, :]
    sv1 = np.linalg.svd(B + gamma * Q, compute_uv=False)
    kappa_bound = (sv0[0] + gamma) / (sv0[-1] + gamma)
    kappa_actual = sv1[0] / (sv1[-1] + 1e-300)
    return {
        "kappa_bound": float(kappa_bound),
        "kappa_actual": float(kappa_actual),
        "margin": float(kappa_bound - kappa_actual),
        "sigma_min_deployed": float(sv1[-1]),
    }


def backward_alignment_score(G, block: int = 64):
    """BAS = 1 - ||G - BD(G)||_F / ||G||_F for a (D, D) gradient second-moment.

    Measures how much of the gradient structure is block-local; high BAS
    means local (block-diagonal) readouts can represent the learning signal.
    G is zero-padded to a multiple of the block size (padding adds no energy).
    """
    G = np.asarray(G, dtype=float)
    D = G.shape[0]
    pad = (-D) % block
    if pad:
        G = np.pad(G, ((0, pad), (0, pad)))
    Dp = G.shape[0]
    nb = Dp // block
    mask = np.kron(np.eye(nb), np.ones((block, block)))
    off = G * (1.0 - mask)
    return float(1.0 - np.linalg.norm(off) / (np.linalg.norm(G) + 1e-12))
