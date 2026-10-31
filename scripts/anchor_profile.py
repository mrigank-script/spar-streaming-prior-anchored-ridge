"""Post-hoc anchor profile: dense gamma grid evaluated from ONE stats pass.

Accumulates forgetting-weighted (A, B, n) once per (seed, drift), then solves
the anchored system for every gamma on the grid with a single closed-form
solve each — the hyperparameter search-budget claim of the paper, plus the
anchor-profile figure (gamma -> tail MSE, with the gamma->infty warm limit).
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

from core import features
from dataio import splits, synthetic

LOOKBACK = 24
TRAIN_END = 2000
TAIL_FRAC = 0.2
ALPHA = 0.999
RIDGE = 0.01
GAMMAS = np.logspace(-3, 2, 30)
FEAT_CFG = features.FeatureConfig(stats=False, lags=(1, 2, 4, 8), fft_bins=0, lru_modes=0)


def prepare(name):
    d = synthetic.load_config(name)
    values = np.concatenate([d["x"].astype(np.float64), d["y"].astype(np.float64)[:, None]], axis=1)
    y_all = values[:, -1]
    ctx_tr, tgt_tr = splits.stream_arrays(TRAIN_END, lookback=LOOKBACK, start=LOOKBACK)
    ctx_st, tgt_st = splits.stream_arrays(len(values), lookback=LOOKBACK, start=TRAIN_END)
    fn = features.make_feature_fn(FEAT_CFG)
    phi_tr = np.asarray(jax.vmap(fn)(jnp.asarray(values[ctx_tr])).block_until_ready())
    phi_st = np.asarray(jax.vmap(fn)(jnp.asarray(values[ctx_st])).block_until_ready())
    mu, sd = phi_tr.mean(0), phi_tr.std(0) + 1e-8
    phi_tr, phi_st = (phi_tr - mu) / sd, (phi_st - mu) / sd
    return phi_tr, y_all[tgt_tr], phi_st, y_all[tgt_st]


def profile(drift, seeds):
    D = None
    curves = []
    t_stats = 0.0
    for seed in seeds:
        phi_tr, y_tr, phi_st, y_st = prepare(synthetic.config_name(32, 10000, drift, seed))
        D = phi_tr.shape[1]
        n_cut = int(len(y_st) * (1.0 - TAIL_FRAC))
        t0 = time.perf_counter()
        A = phi_tr.T @ phi_tr
        B = phi_tr.T @ y_tr
        n = float(len(y_tr))
        for t in range(n_cut):
            A = ALPHA * A + np.outer(phi_st[t], phi_st[t])
            B = ALPHA * B + phi_st[t] * y_st[t]
            n = ALPHA * n + 1.0
        t_stats += time.perf_counter() - t0
        eye = np.eye(D)
        n_tr = float(len(phi_tr))
        W_warm = np.linalg.solve(phi_tr.T @ phi_tr + n_tr * RIDGE * eye, phi_tr.T @ y_tr)
        mse = []
        tail_phi, tail_y = phi_st[n_cut:], y_st[n_cut:]
        for g in GAMMAS:
            W = np.linalg.solve(A + n * (g + RIDGE) * eye, B + n * g * W_warm)
            pred = tail_phi @ W
            mse.append(float(np.mean((pred - tail_y) ** 2)))
        mse_warm = float(np.mean((tail_phi @ W_warm - tail_y) ** 2))
        curves.append((np.array(mse), mse_warm))
    mses = np.stack([c[0] for c in curves])
    warm = np.array([c[1] for c in curves])
    return mses, warm, t_stats


def main() -> None:
    print("backend:", jax.default_backend())
    all_curves = {}
    for drift in ("abrupt", "gradual"):
        t0 = time.perf_counter()
        mses, warm, t_stats = profile(drift, range(10))
        t_all = time.perf_counter() - t0
        print(f"[{drift}] {len(GAMMAS)} gammas x 10 seeds | stats pass {t_stats:.1f}s | "
              f"total {t_all:.1f}s | MSE at g=1e-3: {mses[:, 0].mean():.2f} | "
              f"g=1: {mses[:, GAMMAS.searchsorted(1.0)].mean():.2f} | "
              f"g=100: {mses[:, -1].mean():.2f} | warm limit: {warm.mean():.2f}")
        all_curves[drift] = mses

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ROOT / "results" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"abrupt": "#3b6ea5", "gradual": "#8a5a44"}
    for drift, mses in all_curves.items():
        mean = mses.mean(0)
        std = mses.std(0)
        ax.errorbar(GAMMAS, mean, yerr=std, label=drift, capsize=2, color=colors[drift])
    ax.axhline(warm.mean(), color="gray", ls="--", label="warm limit (gamma->inf)")
    ax.set_xscale("log")
    ax.set_xlabel("anchor strength gamma")
    ax.set_ylabel("tail MSE (last 20% of stream)")
    ax.set_title("Post-hoc anchor profile (one stats pass, 30 gammas)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "anchor_profile.png", dpi=160)
    np.savetxt(ROOT / "results" / "anchor_profile.csv",
                np.column_stack([GAMMAS] + [all_curves[d].mean(0) for d in ("abrupt", "gradual")]),
                delimiter=",", header="gamma,abrupt,gradual", comments="")
    print(f"figure -> {fig_dir / 'anchor_profile.png'}")


if __name__ == "__main__":
    main()
