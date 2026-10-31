"""Metrics suite: compute five diagnostic metrics on existing artifacts.

1. Drift velocity/acceleration on the covariate-shift stream (null property).
2. AROA on the live-trigger variants (adaptation economics).
3. ARR trajectory on a synthetic reanchor stream (anchor telemetry).
4. CCM on the finance cross-moment matrix (deployment certificate + bug catch).
5. BAS on three feature regimes (gradient compressibility).
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_eeg import apply_whitener, bandpass, cov_features, ea_whitener
from core import features
from core.metrics_ext import (
    anchor_reliance,
    aroa,
    backward_alignment_score,
    certified_conditioning,
    drift_kinematics,
)
from dataio import finance, splits, synthetic
from dataio import load_eeg_subject
from runner.align import moment_drift_blocks, pad_blocks
from runner.live import TriggerConfig, run_live

LOOKBACK = 24
TRAIN_END = 2000
SHIFT_START = 3000
FEAT_CFG = features.FeatureConfig(stats=True, lags=(1, 2, 4), fft_bins=0, lru_modes=0)
LIVE_FEAT_CFG = features.FeatureConfig(stats=True, lags=(1, 2, 4), fft_bins=0, lru_modes=0)
STATE_LRU_MODES = 16


def build_shifted_stream():
    d = synthetic.load_config("d32_T10000_gradual_s0.npz")
    values = np.concatenate([d["x"].astype(np.float64), d["y"].astype(np.float64)[:, None]], axis=1)
    w0 = d["w0"].astype(np.float64)
    values[:, -1] = values[:, :32] @ w0
    rng = np.random.default_rng(0)
    scales = np.exp(rng.normal(scale=0.5, size=32))
    biases = rng.normal(scale=1.0, size=32)
    values[SHIFT_START:, :32] = values[SHIFT_START:, :32] * scales + biases
    return values


def demo_drift_kinematics(values):
    lru_params = features.make_lru_params(jax.random.PRNGKey(0), STATE_LRU_MODES, 33)

    def state_fn(win):
        s = features.lru_last_state(win.astype(jnp.float32), lru_params["lambda_diag"], lru_params["B"])
        return jnp.concatenate([jnp.real(s), jnp.imag(s)])

    state_jit = jax.jit(state_fn)
    ctx_tr, _ = splits.stream_arrays(TRAIN_END, lookback=LOOKBACK, start=LOOKBACK)
    ctx_st, _ = splits.stream_arrays(len(values), lookback=LOOKBACK, start=TRAIN_END)
    st_ref = np.asarray(jax.vmap(state_jit)(jnp.asarray(values[ctx_tr])).block_until_ready())
    st_st = np.asarray(jax.vmap(state_jit)(jnp.asarray(values[ctx_st])).block_until_ready())
    onset = (SHIFT_START - TRAIN_END) // 100

    pad_scores = pad_blocks(st_ref, st_st, block=100, rank=8)
    mom_scores, _ = moment_drift_blocks(st_ref, st_st, block=100)
    for tag, scores in (("PAD", pad_scores), ("moment-drift", mom_scores)):
        v, a = drift_kinematics(scores)
        pre_a = np.abs(a[:onset])
        onset_a = np.abs(a[onset: onset + 3])
        z = onset_a.max() / (pre_a.std() + 1e-12)
        print(f"[drift kinematics/{tag}] pre |accel| {pre_a.mean():.4f}±{pre_a.std():.4f} | "
              f"onset |accel| max {onset_a.max():.4f} (z={z:.1f}) | "
              f"onset |vel| {np.abs(v[onset: onset + 3]).max():.4f}/block")


def demo_aroa(values):
    lru_params = features.make_lru_params(jax.random.PRNGKey(0), STATE_LRU_MODES, 33)
    post = slice(SHIFT_START + 500, len(values))
    outs = {}
    for tag, trig, oracle in (
        ("no-trigger", TriggerConfig(use_pad=False, use_moment=False), None),
        ("oracle", TriggerConfig(), SHIFT_START),
        ("live", TriggerConfig(), None),
    ):
        out = run_live(values, TRAIN_END, LIVE_FEAT_CFG, trig, lru_params,
                       lookback=LOOKBACK, oracle_onset_step=oracle)
        err = (out["preds"][post] - out["targets"][post]) ** 2
        n_fires = sum(1 for t in out["triggers"] if t["pad"] is not None)
        outs[tag] = (float(np.nanmean(err)), n_fires)
    e_none, _ = outs["no-trigger"]
    for tag in ("oracle", "live"):
        e_ad, n_act = outs[tag]
        value = aroa(np.full(100, e_none), np.full(100, e_ad), n_actions=n_act)
        print(f"[AROA] {tag}: MSE {e_ad:.3f} with {n_act} adaptation actions -> "
              f"{value:+.3f} MSE-points per action (baseline {e_none:.3f})")


def demo_arr():
    d = synthetic.load_config("d32_T10000_abrupt_s0.npz")
    values = np.concatenate([d["x"].astype(np.float64), d["y"].astype(np.float64)[:, None]], axis=1)
    y_all = values[:, -1]
    ctx_tr, tgt_tr = splits.stream_arrays(TRAIN_END, lookback=LOOKBACK, start=LOOKBACK)
    ctx_st, tgt_st = splits.stream_arrays(len(values), lookback=LOOKBACK, start=TRAIN_END)
    fn = features.make_feature_fn(FEAT_CFG)
    phi_tr = np.asarray(jax.vmap(fn)(jnp.asarray(values[ctx_tr])).block_until_ready())
    phi_st = np.asarray(jax.vmap(fn)(jnp.asarray(values[ctx_st])).block_until_ready())
    mu, sd = phi_tr.mean(0), phi_tr.std(0) + 1e-8
    phi_tr, phi_st = (phi_tr - mu) / sd, (phi_st - mu) / sd
    y_tr, y_st = y_all[tgt_tr], y_all[tgt_st]
    gamma, ridge, alpha = 0.1, 0.01, 0.999
    A = phi_tr.T @ phi_tr
    B = phi_tr.T @ y_tr
    n = float(len(y_tr))
    G = np.linalg.solve(A + n * ridge * np.eye(A.shape[0]), B)
    T = len(y_st)
    marks = [int(T * f) for f in (0.25, 0.5, 0.75, 1.0)]
    arr_traj = []
    for t in range(T):
        A = alpha * A + np.outer(phi_st[t], phi_st[t])
        B = alpha * B + phi_st[t] * y_st[t]
        n = alpha * n + 1.0
        if (t + 1) in marks:
            arr_traj.append((t + 1, anchor_reliance(A, B, n, gamma, ridge, G)))
    print("[ARR] gamma=0.1 trajectory (step, reliance):",
          " ".join(f"({s},{r:.3f})" for s, r in arr_traj))
    arr_end_zero = anchor_reliance(A, B, n, 0.0, ridge, G)
    gamma_sweep = {g: anchor_reliance(A, B, n, g, ridge, G) for g in (0.01, 0.1, 1.0, 10.0)}
    print(f"[ARR] sanity gamma=0 -> {arr_end_zero:.4f} | end-of-stream gamma sweep: " +
          " ".join(f"g={g}:{r:.3f}" for g, r in gamma_sweep.items()))


def demo_ccm():
    close = finance.load_etf_panel()
    rets = np.log(close).diff()
    vol = rets.rolling(20).std()
    feats = [rets.shift(l).to_numpy(float) for l in range(1, 6)] + [vol.to_numpy(float)]
    X = np.stack(feats, axis=-1)
    Y = np.stack([rets.shift(-h).to_numpy(float) for h in (1, 2, 5, 10)], axis=-1)
    valid = np.isfinite(X).all(-1) & np.isfinite(Y).all(-1)
    dates = rets.index
    train = np.asarray(dates <= "2014-12-31") & valid.all(1)
    mu = np.nanmean(X[train], axis=(0, 1))
    sd = np.nanstd(X[train], axis=(0, 1)) + 1e-8
    Xs = (X - mu) / sd
    stream = [t for t in range(len(dates)) if dates[t] > np.datetime64("2014-12-31")]
    B = np.zeros((X.shape[2], Y.shape[2]))
    for count, t in enumerate(stream[:1500]):
        m = valid[t]
        B += Xs[t][m].T @ Y[t][m]
        if count == 749:
            break
    B /= 750.0
    rng4 = np.random.default_rng(4)
    mix = np.zeros((3, 4))
    mix[:, :3] = np.linalg.qr(rng4.normal(size=(3, 3)))[0]
    mix[:, 3] = mix[:, 0] + mix[:, 1]
    B_demo = B[:, :3] @ mix
    sv = np.linalg.svd(B_demo, compute_uv=False)
    gamma = 0.1 * float(sv[0])
    ok = certified_conditioning(B_demo, gamma, mode="padded")
    bad = certified_conditioning(B_demo, gamma, mode="rank-truncated")
    print(f"[CCM] B_demo (6x4, rank-3, sigma=[{sv[0]:.3e}..{sv[-1]:.3e}], gamma={gamma:.3e}): "
          f"padded k_bound={ok['kappa_bound']:.3f} k_actual={ok['kappa_actual']:.3f} "
          f"margin={ok['margin']:+.2e} (certificate tight) | "
          f"rank-truncated k_actual={bad['kappa_actual']:.3e} margin={bad['margin']:+.1f} (bug caught)")


def _mlp_grad_cov(Phi, y, seed=0, hidden=32, steps=80, n_grad=150):
    D = Phi.shape[1]
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    params = {
        "W1": jax.random.normal(k1, (D, hidden)) * np.sqrt(2.0 / D),
        "b1": jnp.zeros(hidden),
        "W2": jax.random.normal(k2, (hidden,)) * np.sqrt(2.0 / hidden),
    }

    def loss_fn(p, x, t):
        h = jnp.tanh(x @ p["W1"] + p["b1"])
        out = h @ p["W2"]
        return jnp.mean((out - t) ** 2)

    grad_loss = jax.jit(jax.grad(loss_fn))
    grad_input = jax.jit(jax.grad(lambda p, x, t: loss_fn(p, x, t), argnums=1))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(Phi), size=32)
        grads = grad_loss(params, jnp.asarray(Phi[idx]), jnp.asarray(y[idx]))
        params = {k: params[k] - 1e-2 * grads[k] for k in params}
    G = np.zeros((D, D))
    for i in range(min(n_grad, len(Phi))):
        g = np.asarray(grad_input(params, jnp.asarray(Phi[i]), float(y[i])))
        G += np.outer(g, g)
    return G / max(1, min(n_grad, len(Phi)))


def demo_bas():
    print("[BAS] two readings: structure = E[phi phi^T] locality (when block readouts suffice); "
          "gradient = MLP input-gradient second-moment locality (when no-BP local learning can match BP)")
    rng_pop = np.random.default_rng(11)
    A_pop = np.zeros((512, 512))
    for i in range(8):
        q, _ = np.linalg.qr(rng_pop.normal(size=(64, 64)))
        s = rng_pop.uniform(0.5, 1.5, 64)
        M = q * s
        A_pop[i * 64:(i + 1) * 64, i * 64:(i + 1) * 64] = M @ M.T
    print(f"  population block-diagonal sanity: structure={backward_alignment_score(A_pop, block=64):.4f}")

    d = synthetic.load_config("d32_T10000_abrupt_s0.npz")
    values = np.concatenate([d["x"].astype(np.float64), d["y"].astype(np.float64)[:, None]], axis=1)
    ctx_tr, tgt_tr = splits.stream_arrays(TRAIN_END, lookback=LOOKBACK, start=LOOKBACK)
    fn = features.make_feature_fn(FEAT_CFG)
    phi_tr = np.asarray(jax.vmap(fn)(jnp.asarray(values[ctx_tr])).block_until_ready())
    G_syn = _mlp_grad_cov(phi_tr, values[:, -1][tgt_tr], seed=2)
    A_syn = phi_tr.T @ phi_tr / len(phi_tr)
    blk = 44 if phi_tr.shape[1] % 44 == 0 else 33
    print(f"  sweep features (D={phi_tr.shape[1]}, blk={blk}): "
          f"structure={backward_alignment_score(A_syn, block=blk):.3f} "
          f"gradient={backward_alignment_score(G_syn, block=blk):.3f}")

    out = load_eeg_subject(dataset="2a", subject=1)
    (X_tr, y_tr), _ = out["train"], out["stream"]
    W_ea = ea_whitener(X_tr)
    ref = cov_features(bandpass(apply_whitener(X_tr, W_ea))).mean(0)
    Fc = cov_features(bandpass(apply_whitener(X_tr, W_ea)), ref)
    mu, sd = Fc.mean(0), Fc.std(0) + 1e-12
    Fz = (Fc - mu) / sd
    G_eeg = _mlp_grad_cov(Fz[:200], y_tr[:200], seed=3)
    A_eeg = Fz.T @ Fz / len(Fz)
    print(f"  EEG cov features (D={Fz.shape[1]}->256, blk=64): "
          f"structure={backward_alignment_score(A_eeg, block=64):.3f} "
          f"gradient={backward_alignment_score(G_eeg, block=64):.3f}")


def main() -> None:
    t0 = time.perf_counter()
    print("backend:", jax.default_backend())
    values = build_shifted_stream()
    demo_drift_kinematics(values)
    demo_aroa(values)
    demo_arr()
    demo_ccm()
    demo_bas()
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
