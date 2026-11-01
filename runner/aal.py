"""AAL streaming runner: closed-form warm start plus online anchored updates.

Statistics are forgetting-weighted cumulative moments
  A = sum_i alpha^{t-i} phi_i phi_i^T,  B = sum_i alpha^{t-i} phi_i y_i^T,
with effective count n = sum_i alpha^{t-i}. The anchored solve is
  W = (A + n (gamma+lambda) I)^{-1} (B + n kappa G),
so the prior's relative strength does not decay with stream length.

Arms:
  warm      : anchored ridge warm start, no online update (cfg.online=False)
  rls       : Sherman-Morrison RLS with forgetting (P maintained)
  reanchor  : periodic exact anchored re-solve on cumulative moments
              anchor in {none, train, stale, spectral}
The stream runs inside one jax.lax.scan; no backpropagation anywhere.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class RunnerConfig(NamedTuple):
    damping: float = 1.0
    ridge: float = 0.01
    gamma: float = 0.01
    exact_map: bool = False
    online: bool = True
    mode: str = "rls"
    anchor: str = "none"
    reanchor_every: int = 100
    dtype: str = "float64"


def _dtype(cfg: RunnerConfig):
    return jnp.float64 if cfg.dtype == "float64" else jnp.float32


def anchored_solve(A, B, n, cfg: RunnerConfig, prior_mean, current_W):
    """Exact anchored solve on cumulative moments (see module docstring)."""
    dtype = _dtype(cfg)
    D = B.shape[0]
    if cfg.anchor == "spectral":
        B_mat = B[:, None] if B.ndim == 1 else B
        U, _, Vt = jnp.linalg.svd(B_mat, full_matrices=False)
        G = (U @ Vt)[:, 0] if B.ndim == 1 else U @ Vt
    elif cfg.anchor == "stale":
        G = current_W
    elif cfg.anchor == "train":
        G = prior_mean
    else:
        G = jnp.zeros_like(B)
    coeff = (cfg.gamma + cfg.ridge) if cfg.exact_map else cfg.gamma
    if cfg.anchor == "none":
        coeff = 0.0
    A_eff = A + n * (cfg.gamma + cfg.ridge) * jnp.eye(D, dtype=dtype)
    B_eff = B + (n * coeff) * G
    W = jnp.linalg.solve(A_eff, B_eff)
    P = jnp.linalg.inv(A_eff)
    return W, P


def warm_start(phi_train, y_train, cfg: RunnerConfig, prior_mean=None):
    """Anchored ridge warm start on cumulative moments.

    Returns (A_inv, W, A, B, n) with A = phi^T phi, B = phi^T y, n = #samples.
    """
    dtype = _dtype(cfg)
    phi = phi_train.astype(dtype)
    y = y_train.astype(dtype)
    A = phi.T @ phi
    B = phi.T @ y
    n = jnp.asarray(float(phi.shape[0]), dtype=dtype)
    prior = prior_mean if prior_mean is not None else jnp.zeros_like(B)
    W, P = anchored_solve(A, B, n, cfg, prior, prior)
    return P, W, A, B, n


def run_stream(phi_stream, y_stream, cfg: RunnerConfig, P0, W0, A0, B0, n0,
               prior_mean, store_weights: bool = True):
    """One predict-then-update pass; returns predictions and weight trajectory."""
    dtype = _dtype(cfg)
    phi_stream = phi_stream.astype(dtype)
    y_stream = y_stream.astype(dtype)
    D = W0.shape[0]
    k = max(int(cfg.reanchor_every), 1)

    def step(carry, pair):
        i, P, W, A, B, n = carry
        phi, y = pair
        pred = phi @ W
        if cfg.online:
            if cfg.mode == "rls":
                K = P @ phi
                denom = cfg.damping + phi @ K
                gain = K / denom
                delta = gain * (y - pred) if W.ndim == 1 else jnp.outer(gain, y - pred)
                W = W + delta
                P = (P - jnp.outer(gain, K)) / cfg.damping
            A = cfg.damping * A + jnp.outer(phi, phi)
            B = cfg.damping * B + (phi * y if B.ndim == 1 else jnp.outer(phi, y))
            n = cfg.damping * n + 1.0
            if cfg.mode == "reanchor":
                def anchor(args):
                    return anchored_solve(args[0], args[1], args[2], cfg, prior_mean, args[3])
                W, P = jax.lax.cond((i + 1) % k == 0, anchor,
                                    lambda args: (args[3], P), (A, B, n, W))
        return (i + 1, P, W, A, B, n), (pred, W)

    init = (jnp.int32(0), P0, W0, A0, B0, n0)
    (_, P, W, A, B, n), (preds, weights) = jax.lax.scan(step, init, (phi_stream, y_stream))
    if not store_weights:
        weights = jnp.empty((0, D), dtype=dtype)
    return preds, weights, P, W


def make_stream_fn(cfg: RunnerConfig, store_weights: bool = True):
    def fn(phi_stream, y_stream, P0, W0, A0, B0, n0, prior_mean):
        return run_stream(phi_stream, y_stream, cfg, P0, W0, A0, B0, n0, prior_mean,
                          store_weights=store_weights)
    return jax.jit(fn)


def run_method(phi_tr, y_tr, phi_st, y_st, cfg: RunnerConfig, prior_mean=None):
    """Full arm: warm start + stream. Returns (preds, weights, P, W)."""
    P0, W0, A0, B0, n0 = warm_start(phi_tr, y_tr, cfg, prior_mean)
    if prior_mean is None:
        prior_mean = W0 if cfg.anchor in ("train", "stale") else jnp.zeros_like(W0)
    stream_fn = make_stream_fn(cfg)
    return stream_fn(phi_st, y_st, P0, W0, A0, B0, n0, prior_mean)
