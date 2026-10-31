"""All-channel covariate shift: alignment recovery + PAD on the LRU state."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import features
from dataio import splits, synthetic
from runner import aal
from runner.align import (
    align_windows,
    moment_drift_blocks,
    pad_blocks,
    pad_blocks_cov,
    trigger_metrics,
)

LOOKBACK = 24
TRAIN_END = 2000
SHIFT_START = 3000
CAL_LEN = 500
LRU_MODES = 16
FEAT_CFG = features.FeatureConfig(stats=True, lags=(1, 2, 4), fft_bins=0, lru_modes=0)


def main() -> None:
    print("backend:", jax.default_backend(), "| x64:", jax.config.jax_enable_x64)
    d = synthetic.load_config("d32_T10000_gradual_s0.npz")
    values = np.concatenate([d["x"].astype(np.float64), d["y"].astype(np.float64)[:, None]], axis=1)
    w0 = d["w0"].astype(np.float64)
    values[:, -1] = values[:, :32] @ w0
    rng = np.random.default_rng(0)
    scales = np.exp(rng.normal(scale=0.5, size=32))
    biases = rng.normal(scale=1.0, size=32)
    values[:, -1] = values[:, :32] @ w0
    values[SHIFT_START:, :32] = values[SHIFT_START:, :32] * scales + biases
    y_all = values[:, -1]

    ctx_tr, tgt_tr = splits.stream_arrays(TRAIN_END, lookback=LOOKBACK, start=LOOKBACK)
    ctx_st, tgt_st = splits.stream_arrays(len(values), lookback=LOOKBACK, start=TRAIN_END)
    win_tr, win_st = values[ctx_tr], values[ctx_st]
    y_st = jnp.asarray(y_all[tgt_st])
    y_np = np.asarray(y_st)
    shift_rel = SHIFT_START - TRAIN_END
    cal = win_st[shift_rel:shift_rel + CAL_LEN]

    feature_fn = features.make_feature_fn(FEAT_CFG)
    phi_tr = jax.vmap(feature_fn)(jnp.asarray(win_tr)).block_until_ready()
    mu, sd = phi_tr.mean(0), phi_tr.std(0) + 1e-8
    phi_tr_s = (phi_tr - mu) / sd

    cfg = aal.RunnerConfig(online=False)
    pre = slice(0, shift_rel)
    post = slice(shift_rel + CAL_LEN, len(y_np))

    phi_raw = (jax.vmap(feature_fn)(jnp.asarray(win_st)) - mu) / sd
    preds, _, _, _ = aal.run_method(phi_tr_s, jnp.asarray(y_all[tgt_tr]), phi_raw, y_st, cfg)
    preds = np.asarray(preds)
    print(f"no-align    MSE pre={np.mean((preds[pre] - y_np[pre]) ** 2):.4f} "
          f"post={np.mean((preds[post] - y_np[post]) ** 2):.4f}")
    for tag, method in (("cov-match", "covariance_matching"), ("diag-match", "diagonal")):
        apply_align, _ = align_windows(win_tr, cal, method=method)
        phi = (jax.vmap(feature_fn)(jnp.asarray(apply_align(win_st))) - mu) / sd
        preds, _, _, _ = aal.run_method(phi_tr_s, jnp.asarray(y_all[tgt_tr]), phi, y_st, cfg)
        preds = np.asarray(preds)
        print(f"{tag:<11} MSE pre={np.mean((preds[pre] - y_np[pre]) ** 2):.4f} "
              f"post={np.mean((preds[post] - y_np[post]) ** 2):.4f}")

    lru_params = features.make_lru_params(jax.random.PRNGKey(0), LRU_MODES, 33)

    def state_fn(win):
        s = features.lru_last_state(win, lru_params["lambda_diag"], lru_params["B"])
        return jnp.concatenate([jnp.real(s), jnp.imag(s)])

    state_jit = jax.jit(state_fn)
    st_ref = np.asarray(jax.vmap(state_jit)(jnp.asarray(win_tr)).block_until_ready())
    st_st = np.asarray(jax.vmap(state_jit)(jnp.asarray(win_st)).block_until_ready())

    onset_block = shift_rel // 100
    row_scores = pad_blocks(st_ref, st_st, block=100, rank=8)
    cov_scores = pad_blocks_cov(st_ref, st_st, block=100, rank=8)
    mu_drift, cov_drift = moment_drift_blocks(st_ref, st_st, block=100)
    for tag, scores in (
        ("pad-rows", row_scores),
        ("pad-cov", cov_scores),
        ("mom-mean", mu_drift),
        ("mom-cov", cov_drift),
    ):
        thresh, far, delay = trigger_metrics(scores, onset_block, k=3.0)
        print(f"{tag:<9} pre={scores[:onset_block].mean():.4f} "
              f"post={scores[onset_block:onset_block + 5].mean():.4f} "
              f"threshold={thresh:.4f} FAR={far:.2f} delay={delay:.1f} blocks")


if __name__ == "__main__":
    main()
