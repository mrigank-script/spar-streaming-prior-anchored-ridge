"""Regenerate the prediction-unit SPAR-Pull and SPAR-EG+ grid rows.

The legacy appended-grid driver cached the parameter-unit (`spar_kf2`)
trajectory for the SPAR-EG+ pool.  This script recomputes both rows with
`prediction_units=True` for every main-grid and stress key and writes them
to `results/factorial_pu_refresh.csv`.  Merging that file last into
`factorial_extras.csv` makes the prediction-unit rows authoritative for
`aggregate_factorial.py` (which keeps the last duplicate).

Usage:
  py scripts/regen_spar_eg_plus.py --workers 6
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from revision_runs import (GRID_COLS, MAIN_GRID, STRESS, T_WARM,  # noqa: E402
                           D, T, _row, _stream)
from rls_baselines import run_spar_eg, run_spar_kf  # noqa: E402

RES = ROOT / "results"
OUT = RES / "factorial_pu_refresh.csv"


def run_one(task):
    spectrum, drift, prior, noise, seed = task
    t0 = time.perf_counter()
    x, y, G, y_te, regs, t1p = _stream(spectrum, drift, prior, noise, seed)
    base = dict(spectrum=spectrum, drift=drift, prior=prior, noise=noise,
                seed=seed)
    meta = lambda m: {**base, "method": m, "D": D, "T": T}
    preds, _, _ = run_spar_kf(
        x, y, T_WARM, G_init=G, refresh_every=10, resid_fast=30,
        rho_drift_floor=0.02, debias_variance_floor=True,
        prediction_units=True, return_diag=True)
    rows = [_row(meta("spar_pull"), preds, y_te, regs, t1p,
                 time.perf_counter() - t0)]
    preds_egp = run_spar_eg(x, y, T_WARM, G_init=G, extra_preds=preds)
    rows.append(_row(meta("spar_eg_plus"), preds_egp, y_te, regs, t1p,
                     time.perf_counter() - t0))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # MAIN_GRID/STRESS are (drift, spectrum, prior, noise); the runner takes
    # (spectrum, drift, prior, noise, seed).
    tasks = [(spec, drift, prior, noise, seed)
             for (drift, spec, prior, noise) in MAIN_GRID
             for seed in range(5)]
    tasks += [(spec, drift, prior, noise, seed)
              for (drift, spec, prior, noise) in STRESS
              for seed in range(5)]
    if args.limit:
        tasks = tasks[:args.limit]

    print(f"pending={len(tasks)} workers={args.workers}", flush=True)
    if OUT.exists():
        OUT.unlink()
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            rows = fut.result()
            pd.DataFrame(rows).reindex(columns=GRID_COLS).to_csv(
                OUT, mode="a", header=not OUT.exists(), index=False)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] {time.perf_counter()-t0:.0f}s",
                      flush=True)
    df = pd.read_csv(OUT)
    main = df[df.drift.isin(["abrupt", "both", "covrot"])
              & (df.noise == "gauss")]
    for m, g in df.groupby("method"):
        x = g.mse_overall.to_numpy()
        print(f"{m:14s} all n={len(x)} mean={x.mean():.4f} "
              f"worst={x.max():.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()