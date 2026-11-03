"""Real-stream external validity: random-client population and multi-step horizon.

Extends the electricity benchmark (which reports three percentile clients) to
(a) a seeded random sample of clients, so the untuned transfer is not specific
to the three reported percentiles, and (b) an h-step-ahead target for
h in {1, 6, 12, 24}, which probes the lag/variance tradeoff across horizons.
The arms and grid-default control parameters are unchanged.

Writes results/stream_random_clients.csv and results/stream_horizon.csv.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_stream_real import GRID, T_END, T_WARM, load_frames  # noqa: E402
from rls_baselines import (run_kf_est, run_leaky_batch, run_spar_eg,  # noqa: E402
                           run_spar_kf)

N_RANDOM = 6
HORIZONS = (1, 12)


def build_h(vals, client, h):
    feats = vals[:-h]
    target = vals[h:, client]
    hour = (np.arange(len(feats)) % 24).astype(np.float64)
    extra = np.stack([np.sin(2 * np.pi * hour / 24),
                      np.cos(2 * np.pi * hour / 24)], axis=1)
    X = np.concatenate([feats, extra], axis=1)
    mu = X[:T_WARM].mean(0)
    sd = X[:T_WARM].std(0)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd
    X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    y = target - target[:T_WARM].mean()
    return X[:T_END], y[:T_END]


def arms(x, y):
    out = {}
    preds, _ = run_spar_kf(x, y, T_WARM, G_init=None, **GRID)
    out["SPAR-Pull"] = preds
    for lam in (0.999, 0.99):
        out[f"leaky-info[{lam}]"] = run_leaky_batch(x, y, T_WARM, lam=lam)
    out["kf-slow[q=1e-05]"] = run_kf_est(x, y, T_WARM, q=1e-5, eta_R=5e-5)
    return out


def mse(preds, y):
    return float(np.mean((preds - y[T_WARM:]) ** 2))


def main() -> None:
    vals, picks = load_frames()
    rng = np.random.default_rng(0)
    clients = sorted(int(c) for c in rng.choice(vals.shape[1], N_RANDOM,
                                                replace=False))
    print(f"random clients: {clients}", flush=True)

    rrows = []
    t_all = time.perf_counter()
    for client in clients:
        x, y = build_h(vals, client, 1)
        task = arms(x, y)
        base_names = [k for k in task if not k.startswith("SPAR")]
        best_base = min(mse(task[k], y) for k in base_names)
        for name, preds in task.items():
            m = mse(preds, y)
            rrows.append(dict(client=client, D=x.shape[1], horizon=1,
                              method=name, mse=m))
        sp_pull = mse(task["SPAR-Pull"], y)
        print(f"client {client}: SPAR-Pull={sp_pull:.5f} "
              f"best_base={best_base:.5f}", flush=True)
    pd.DataFrame(rrows).to_csv(ROOT / "results" / "stream_random_clients.csv",
                               index=False)

    df = pd.DataFrame(rrows)
    piv = df.pivot_table(index="client", columns="method", values="mse")
    base_cols = [c for c in piv.columns if not c.startswith("SPAR")]
    wins = int((piv["SPAR-Pull"] <= piv[base_cols].min(axis=1)).sum())
    print("\nrandom-client means:")
    print(piv.mean().round(5).to_string())
    print(f"SPAR-Pull <= best baseline on {wins}/{len(piv)} clients")

    if "--horizon" in sys.argv:
        hrows = []
        for h in HORIZONS:
            for client in picks:
                x, y = build_h(vals, client, h)
                task = arms(x, y)
                for name, preds in task.items():
                    hrows.append(dict(client=int(client), D=x.shape[1],
                                      horizon=h, method=name,
                                      mse=mse(preds, y)))
            print(f"horizon {h} done [{time.perf_counter()-t_all:.0f}s]",
                  flush=True)
        pd.DataFrame(hrows).to_csv(ROOT / "results" / "stream_horizon.csv",
                                   index=False)
        print("\nhorizon sweep (mean over the three percentile clients):")
        hp = pd.DataFrame(hrows).pivot_table(index="horizon", columns="method",
                                             values="mse")
        print(hp.round(5).to_string())
    print("\nsaved results/stream_random_clients.csv")


if __name__ == "__main__":
    main()