"""Append the SPAR-Pull arm to the completed factorial grid (same streams,
same seeds — deterministic generator guarantees identical data)."""

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

from rls_baselines import make_prior, make_stream_fac, run_spar_kf  # noqa: E402
from synthetic_factorial import phases_for  # noqa: E402

RES = ROOT / "results"

MAIN = [(d, s, p, "gauss")
        for d in ("abrupt", "covrot", "both")
        for s in ("iso", "decay3", "rankdef")
        for p in ("oracle", "warm", "adversarial")]
EXTRA = [("exdeath", "decay3", "warm", "gauss"), ("osc", "iso", "warm", "gauss"),
         ("covrot", "decay3", "warm", "t3"), ("none", "iso", "warm", "gauss")]


def append_arm(tag, combos, seeds=5, D=128, T=10000, T_warm=2000):
    out_csv = RES / f"factorial_{tag}.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        have = prev[prev.method == "spar_kf"]
        done = set(zip(have["spectrum"], have["drift"], have["prior"],
                       have["noise"], have["seed"]))
    n_new = 0
    t0 = time.perf_counter()
    rows = []
    for drift, spectrum, prior, noise in combos:
        for seed in range(seeds):
            if (spectrum, drift, prior, noise, seed) in done:
                continue
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum,
                drift=drift, noise=noise)
            G_init = make_prior(prior, x, y, w_true, t1, T_warm, Q0, lam0)
            y_te = y[T_warm:]
            n_te = len(y_te)
            regs = phases_for(drift, t1 - T_warm, t2 - T_warm, n_te)
            preds, _ = run_spar_kf(
                x, y, T_warm, G_init=G_init, refresh_every=10,
                resid_fast=30, rho_drift_floor=0.02)
            row = {"spectrum": spectrum, "drift": drift, "prior": prior,
                   "noise": noise, "seed": seed, "method": "spar_kf",
                   "D": D, "T": T,
                   "wall_s": time.perf_counter() - t0}
            for reg_name, reg in regs.items():
                row[f"mse_{reg_name}"] = float(
                    np.mean((preds[reg] - y_te[reg]) ** 2)) \
                    if reg.stop > reg.start else float("nan")
            row["mse_overall"] = float(np.mean((preds - y_te) ** 2))
            row["recovery"] = float("nan")
            row["diverged"] = bool(row["mse_overall"] > 1e3
                                   or not np.all(np.isfinite(preds)))
            rows.append(row)
            n_new += 1
            if len(rows) >= 25:
                pd.DataFrame(rows).to_csv(out_csv, mode="a",
                                          header=not out_csv.exists(),
                                          index=False)
                rows = []
                print(f"[{tag}] {drift}/{spectrum}/{prior}/{noise} s{seed} done",
                      flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, mode="a",
                                  header=not out_csv.exists(), index=False)
    print(f"[{tag}] appended {n_new} spar_kf rows "
          f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    for tag, combos in (("d_abrupt", MAIN[:9]), ("d_covrot", MAIN[9:18]),
                        ("d_both", MAIN[18:]), ("extras", EXTRA)):
        append_arm(tag, combos)
