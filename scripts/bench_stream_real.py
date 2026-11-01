"""Real high-dimensional stream: hourly electricity load (D = 321 clients).

A real stream longer than T = 5000 and wider than D = 128, run with the
*unmodified* grid-default control parameters: no stream-specific window
range and no per-configuration tuning.

Design.  Features at hour t are the log1p loads of all 321 clients
(standardized on the warm-up) plus an intercept and hour-of-day sine/cosine
(D = 324); the target is the log1p load of one held-out client at t + 1,
centered on its warm-up mean.  Three target clients (25th, 50th, 75th
percentile of mean load) give three tasks; every arm sees the same stream.
Baselines are swept over their own hyperparameter and reported at the best
setting (oracle-tuned, which favours them); the SPAR arms use the grid
defaults with no sweep.

Rows are written to CSV after every arm, so a partial run is still usable.
"""

import argparse
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rls_baselines import (run_kf_est, run_leaky_batch, run_spar_eg,  # noqa: E402
                           run_spar_kf)

T_WARM = 2000
T_END = 8000
N_CLIENTS_KEEP = (0.25, 0.50, 0.75)

# Grid defaults (unchanged from the paper's hyperparameter table).
GRID = dict(refresh_every=10, resid_fast=30, rho_drift_floor=0.02,
            debias_variance_floor=True, prediction_units=True,
            w_min=5e-4, w_max=0.05)

OUT = ROOT / "results" / "stream_electricity.csv"


def load_frames():
    from dataio.ts import resolve_path
    raw = pd.read_csv(resolve_path("Electricity"),
                      header=None).to_numpy(np.float64)
    vals = np.log1p(raw)
    order = np.argsort(vals.mean(0))
    picks = [int(order[int(q * (len(order) - 1))]) for q in N_CLIENTS_KEEP]
    return vals, picks


def build(vals, client):
    feats = vals[:-1]                      # feature hour t
    target = vals[1:, client]              # target hour t + 1
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


def save(rows):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", default="")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    vals, picks = load_frames()
    want = ([int(c) for c in args.clients.split(",") if c.strip()]
            if args.clients else picks)
    rows = []
    if not args.fresh and OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    print(f"clients: {want}", flush=True)

    t_all = time.perf_counter()
    for client in want:
        x, y = build(vals, client)
        D = x.shape[1]
        print(f"\n[elec] client {client} D={D} T_eval={len(y) - T_WARM}",
              flush=True)

        def record(method, preds, extra=None):
            err = (preds - y[T_WARM:]) ** 2
            row = dict(client=client, D=D, method=method,
                       T_eval=int(len(y) - T_WARM),
                       mse=float(np.mean(err)),
                       mse_last2000=float(np.mean(err[-2000:])))
            if extra:
                row.update(extra)
            rows.append(row)
            save(rows)
            print(f"  {method:26s} mse={row['mse']:.5f} "
                  f"last2000={row['mse_last2000']:.5f}", flush=True)

        t0 = time.perf_counter()
        preds, _, _ = run_spar_kf(x, y, T_WARM, G_init=None,
                                  return_diag=True, **GRID)
        record("SPAR-Pull", preds, {"wall_s": time.perf_counter() - t0})

        t0 = time.perf_counter()
        preds_egp = run_spar_eg(x, y, T_WARM, G_init=None, extra_preds=preds)
        record("SPAR-EG+", preds_egp, {"wall_s": time.perf_counter() - t0})

        for lam in (0.999, 0.995, 0.99):
            t0 = time.perf_counter()
            p = run_leaky_batch(x, y, T_WARM, lam=lam)
            record(f"leaky-info[{lam}]", p,
                   {"wall_s": time.perf_counter() - t0})

        for q in (1e-4, 1e-5):
            t0 = time.perf_counter()
            p = run_kf_est(x, y, T_WARM, q=q, eta_R=5e-5)
            record(f"kf-slow[q={q:g}]", p, {"wall_s": time.perf_counter() - t0})

        print(f"  [task wall {time.perf_counter() - t_all:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    print()
    print(df.groupby("method")[["mse", "mse_last2000"]].mean()
          .round(5).to_string())
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()