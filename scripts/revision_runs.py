"""Revision runs: prediction-unit SPAR, expanded baseline grids, ablations.

Usage:
  py scripts/revision_runs.py --suite grid --workers 6
  py scripts/revision_runs.py --suite ablation --workers 6
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rls_baselines import (make_prior, make_stream_fac, run_adam,  # noqa: E402
                           run_arow, run_df, run_spar_eg, run_spar_kf)
from synthetic_factorial import phases_for, recovery_time  # noqa: E402

T_WARM, D, T = 2000, 128, 10000
RES = ROOT / "results"

MAIN_GRID = [(d, s, p, "gauss")
             for d in ("abrupt", "covrot", "both")
             for s in ("iso", "decay3", "rankdef")
             for p in ("oracle", "warm", "adversarial")]
STRESS = [("exdeath", "decay3", "warm", "gauss"),
          ("osc", "iso", "warm", "gauss"),
          ("covrot", "decay3", "warm", "t3"),
          ("none", "iso", "warm", "gauss")]
ABL_CFG = [("abrupt", "decay3", "warm", "gauss"),
           ("covrot", "iso", "warm", "gauss"),
           ("abrupt", "iso", "adversarial", "gauss"),
           ("osc", "iso", "warm", "gauss")]
BASE_METHODS = ["df[0.999]", "df[0.997]", "df[0.995]", "df[0.99]",
                "adam[0.001]", "adam[0.0003]",
                "arow[0.01]", "arow[0.1]", "arow[0.5]"]
ABL_VARIANTS = {
    "full": {},
    "fixed_w": dict(w_min=5e-4, w_max=5e-4),
    "no_cusum": dict(use_cusum=False),
    "no_ref": dict(use_ref=False),
    "no_refresh": dict(refresh=False),
    "fixed_rho": dict(adaptive_rho=False),
    "gamma=0.5": dict(gamma=0.5),
    "no_vfloor": dict(debias_variance_floor=False),
}


def _stream(spectrum, drift, prior, noise, seed):
    x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
        seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum, drift=drift,
        noise=noise)
    G = make_prior(prior, x, y, w_true, t1, T_WARM, Q0, lam0)
    y_te = y[T_WARM:]
    regs = phases_for(drift, t1 - T_WARM, t2 - T_WARM, len(y_te))
    return x, y, G, y_te, regs, t1 - T_WARM


def _row(meta, preds, y_te, regs, t1p, wall=0.0, include_early=False):
    row = dict(meta)
    for name, reg in regs.items():
        row[f"mse_{name}"] = (float(np.mean((preds[reg] - y_te[reg]) ** 2))
                              if reg.stop > reg.start else float("nan"))
    row["mse_overall"] = float(np.mean((preds - y_te) ** 2))
    if include_early:
        row["mse_early"] = float(np.mean(
            (preds[t1p:t1p + 500] - y_te[t1p:t1p + 500]) ** 2)) \
            if t1p + 500 <= len(y_te) else float("nan")
    row["recovery"] = (recovery_time(preds, y_te, t1p, row.get("mse_pre"))
                       if regs.get("dur", slice(0, 0)).stop >
                       regs.get("dur", slice(0, 0)).start else float("nan"))
    row["diverged"] = bool(row["mse_overall"] > 1e3
                           or not np.all(np.isfinite(preds)))
    row["wall_s"] = wall
    return row


def run_grid(task):
    spectrum, drift, prior, noise, seed = task
    t0 = time.perf_counter()
    x, y, G, y_te, regs, t1p = _stream(spectrum, drift, prior, noise, seed)
    base = dict(spectrum=spectrum, drift=drift, prior=prior, noise=noise,
                seed=seed)
    meta = lambda m: {**base, "method": m, "D": D, "T": T}
    rows = []
    preds, _, _ = run_spar_kf(
        x, y, T_WARM, G_init=G, refresh_every=10, resid_fast=30,
        rho_drift_floor=0.02, debias_variance_floor=True,
        prediction_units=True, return_diag=True)
    rows.append(_row(meta("spar_pull"), preds, y_te, regs,
                     t1p, time.perf_counter() - t0))
    preds_eg = run_spar_eg(x, y, T_WARM, G_init=G)
    rows.append(_row(meta("spar_eg"), preds_eg, y_te,
                     regs, t1p, time.perf_counter() - t0))
    preds_egp = run_spar_eg(x, y, T_WARM, G_init=G, extra_preds=preds)
    rows.append(_row(meta("spar_eg_plus"), preds_egp, y_te,
                     regs, t1p, time.perf_counter() - t0))
    for name in BASE_METHODS:
        t1 = time.perf_counter()
        if name.startswith("df["):
            lam = float(name[3:-1])
            p = run_df(x, y, T_WARM, lam_f=lam, rho=0.1)
        elif name.startswith("adam["):
            lr = float(name[5:-1])
            p = run_adam(x, y, T_WARM, lr=lr)
        else:
            r = float(name[5:-1])
            p = run_arow(x, y, T_WARM, r=r)
        rows.append(_row(meta(name), p, y_te, regs, t1p,
                         time.perf_counter() - t1))
    return drift, rows


def run_stress(task):
    spectrum, drift, prior, noise, seed = task
    t0 = time.perf_counter()
    x, y, G, y_te, regs, t1p = _stream(spectrum, drift, prior, noise, seed)
    base = dict(spectrum=spectrum, drift=drift, prior=prior, noise=noise,
                seed=seed)
    meta = lambda m: {**base, "method": m, "D": D, "T": T}
    rows = []
    preds, _, _ = run_spar_kf(
        x, y, T_WARM, G_init=G, refresh_every=10, resid_fast=30,
        rho_drift_floor=0.02, debias_variance_floor=True,
        prediction_units=True, return_diag=True)
    rows.append(_row(meta("spar_pull"), preds, y_te, regs,
                     t1p, time.perf_counter() - t0))
    preds_egp = run_spar_eg(x, y, T_WARM, G_init=G, extra_preds=preds)
    rows.append(_row(meta("spar_eg_plus"), preds_egp, y_te,
                     regs, t1p, time.perf_counter() - t0))
    return "extras", rows


def run_abl(task):
    drift, spectrum, prior, noise, seed, variant = task
    x, y, G, y_te, regs, t1p = _stream(spectrum, drift, prior, noise, seed)
    kw = ABL_VARIANTS[variant]
    preds = run_spar_kf(
        x, y, T_WARM, G_init=G,
        refresh_every=kw.get("refresh_every", 10),
        resid_fast=kw.get("resid_fast", 30),
        rho_drift_floor=0.02, debias_variance_floor=kw.get(
            "debias_variance_floor", True),
        prediction_units=True,
        **{k: v for k, v in kw.items()
           if k not in ("refresh_every", "resid_fast",
                        "debias_variance_floor")})[0]
    row = _row(dict(spectrum=spectrum, drift=drift, prior=prior, noise=noise,
                    seed=seed, method=variant, D=D, T=T), preds, y_te, regs,
                t1p, include_early=True)
    return "abl", [row]


GRID_COLS = ["spectrum", "drift", "prior", "noise", "seed", "method", "D", "T",
             "wall_s", "mse_pre", "mse_dur", "mse_post", "mse_overall",
             "recovery", "diverged"]
ABL_COLS = ["spectrum", "drift", "prior", "noise", "seed", "method", "D", "T",
            "mse_pre", "mse_dur", "mse_post", "mse_overall", "mse_early",
            "recovery", "diverged", "wall_s"]


def append(dest, rows):
    path = RES / dest
    cols = ABL_COLS if dest == "ablation_spar_pull.csv" else GRID_COLS
    pd.DataFrame(rows).reindex(columns=cols).to_csv(
        path, mode="a", header=not path.exists(), index=False)


def existing_keys(suite):
    files = [f"factorial_d_{d}.csv" for d in ("abrupt", "covrot", "both")]
    files += ["factorial_extras.csv", "factorial_pu_refresh.csv",
              "ablation_spar_pull.csv"]
    frames = []
    for f in files:
        p = RES / f
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return set()
    df = pd.concat(frames, ignore_index=True)
    if suite == "ablation":
        need = set(ABL_VARIANTS)
        out = set()
        for key, g in df.groupby(["drift", "spectrum", "prior", "seed"]):
            if need <= set(g["method"]):
                out.add(tuple(key))
        return out
    need_main = {"spar_pull", "spar_eg", "spar_eg_plus", *BASE_METHODS}
    need_stress = {"spar_pull", "spar_eg_plus"}
    out = set()
    for key, g in df.groupby(["spectrum", "drift", "prior", "noise", "seed"]):
        stress = key[3] == "t3" or key[1] in ("exdeath", "osc", "none")
        if (need_stress if stress else need_main) <= set(g["method"]):
            out.add(tuple(key))
    return out


def dispatch_grid(t):
    kind, *rest = t
    return run_stress(tuple(rest)) if kind == "stress" else run_grid(tuple(rest))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["grid", "ablation"], default="grid")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    done = existing_keys(args.suite)
    tasks = []
    if args.suite == "grid":
        for drift, spectrum, prior, noise in MAIN_GRID:
            for seed in range(5):
                key = (spectrum, drift, prior, noise, seed)
                if key not in done:
                    tasks.append(("grid", spectrum, drift, prior, noise, seed))
        for drift, spectrum, prior, noise in STRESS:
            for seed in range(5):
                key = (spectrum, drift, prior, noise, seed)
                if key not in done:
                    tasks.append(("stress", spectrum, drift, prior, noise,
                                  seed))

        fn = dispatch_grid
    else:
        for cfg in ABL_CFG:
            for seed in range(5):
                for v in ABL_VARIANTS:
                    key = (cfg[0], cfg[1], cfg[2], seed, v)
                    if key not in done:
                        tasks.append((*cfg, seed, v))
        fn = run_abl

    if args.limit:
        tasks = tasks[:args.limit]
    print(f"suite={args.suite} pending={len(tasks)} done_keys={len(done)} "
          f"workers={args.workers}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fn, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            dest, rows = fut.result()
            if args.suite == "ablation":
                append("ablation_spar_pull.csv", rows)
            else:
                append(f"factorial_d_{dest}.csv" if dest != "extras"
                       else "factorial_extras.csv", rows)
            print(f"[{i}/{len(tasks)}] {futs[fut]} -> {dest} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"done ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
