"""Fixed feature builder for streaming windows.

Window convention: (L, C) time-major. Blocks:
  stats : per-channel mean/std/min/max/first/last
  lags  : per-channel values at selected lags
  fft   : per-channel magnitudes of fixed low-frequency rfft bins
  lru   : fixed complex LRU reservoir last state (real + imag)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from core.lru import init_lru_params, lru_combine


class FeatureConfig(NamedTuple):
    stats: bool = True
    lags: tuple = (1, 2, 4, 8)
    fft_bins: int = 4
    lru_modes: int = 0


def feature_dim(cfg: FeatureConfig, n_channels: int) -> int:
    dim = 6 * n_channels if cfg.stats else 0
    dim += len(cfg.lags) * n_channels
    dim += cfg.fft_bins * n_channels
    dim += 2 * cfg.lru_modes
    return dim


def _stats(window):
    return jnp.concatenate(
        [window.mean(0), window.std(0), window.min(0), window.max(0), window[0], window[-1]]
    )


def _lags(window, lags):
    length = window.shape[0]
    return jnp.concatenate([window[length - 1 - lag] for lag in lags])


def _fft(window, bins):
    spectrum = jnp.abs(jnp.fft.rfft(window, axis=0))
    return spectrum[1:bins + 1].reshape(-1)


def lru_last_state(window, lambda_diag, B):
    """Complex last hidden state of the fixed LRU reservoir over a (L, C) window."""
    h_init = jnp.einsum("di,li->ld", B.astype(jnp.float32), window.astype(jnp.float32)).astype(jnp.complex64)
    lambdas = jnp.broadcast_to(lambda_diag.astype(jnp.complex64)[None, :], h_init.shape)
    h_all, _ = jax.lax.associative_scan(lru_combine, (h_init, lambdas), axis=0)
    return h_all[-1]


def build_features(window, cfg: FeatureConfig, lru_params=None):
    blocks = []
    if cfg.stats:
        blocks.append(_stats(window))
    if cfg.lags:
        assert max(cfg.lags) < window.shape[0], (
            f"max lag {max(cfg.lags)} must be < window length {window.shape[0]}"
        )
        blocks.append(_lags(window, cfg.lags))
    if cfg.fft_bins > 0:
        blocks.append(_fft(window, cfg.fft_bins))
    if cfg.lru_modes > 0:
        state = lru_last_state(window, lru_params["lambda_diag"], lru_params["B"])
        blocks.append(jnp.concatenate([jnp.real(state), jnp.imag(state)]))
    return jnp.concatenate(blocks)


def make_feature_fn(cfg: FeatureConfig, lru_params=None, dtype=jnp.float64):
    def fn(window):
        return build_features(window.astype(dtype), cfg, lru_params).astype(dtype)
    return jax.jit(fn)


def make_lru_params(key, modes: int, n_channels: int):
    return init_lru_params(key, modes, n_channels)
