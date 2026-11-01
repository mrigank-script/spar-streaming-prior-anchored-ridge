"""Mismatched-generator configs — testing the estimators outside the theory's
model class.

Three mismatches:
  nonlinear: y = tanh(x·w*) + ε — the linear head is misspecified
  covshift:  feature covariance doubles mid-stream (scale mismatch)
  hetnoise:  observation-noise variance 0.01 -> 0.05 from t1 (pure noise
             scale change, no target drift — isolates the noise-estimation
             horizons of the residual-driven arms)

Question: do the estimators degrade gracefully, or catastrophically?
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

from rls_baselines import (make_stream_fac, make_prior, run_kf, run_kf_est,  # noqa: E402
                           run_leaky_batch, run_rls, run_spar_eg,
                           run_spar_kf)

RES = ROOT / "results"

MISMATCH_CONFIGS = [
    ("abrupt", "decay3", "warm", "gauss", "nonlinear"),
    ("abrupt", "iso", "warm", "gauss", "nonlinear"),
    ("abrupt", "decay3", "warm", "gauss", "covshift"),
    ("none", "decay3", "warm", "gauss", "nonlinear"),
    ("gradual", "decay3", "warm", "gauss", "nonlinear"),
    ("none", "decay3", "warm", "gauss", "hetnoise"),
    ("none", "iso", "warm", "gauss", "hetnoise"),
]


def run_mismatch_arms(x, y, t1, t2, T_warm, G_init):
    out = {}
    kf_preds, _ = run_spar_kf(
        x, y, T_warm, G_init=G_init, refresh_every=10, resid_fast=30,
        rho_drift_floor=0.02, debias_variance_floor=True, prediction_units=True)
    out["spar_pull"] = kf_preds
    out["spar_eg"] = run_spar_eg(x, y, T_warm, G_init=G_init)
    out["spar_eg_plus"] = run_spar_eg(x, y, T_warm, G_init=G_init,
                                       extra_preds=kf_preds)
    out["kf_est"] = run_kf_est(x, y, T_warm, q=1e-4)
    out["kf_est_slow[q=1e-4]"] = run_kf_est(x, y, T_warm, q=1e-4, eta_R=5e-5)
    out["kf_true[q=1e-4]"] = run_kf(x, y, T_warm, q=1e-4, R=0.01)
    out["leaky[0.997]"] = run_rls(x, y, T_warm, lam=0.997)
    out["leaky[0.995]"] = run_rls(x, y, T_warm, lam=0.995)
    out["leaky_batch[0.99]"] = run_leaky_batch(x, y, T_warm, lam=0.99)
    return out


def main() -> None:
    T_warm, D, T = 2000, 128, 10000
    out_csv = RES / "factorial_mismatch.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["spectrum"], prev["drift"], prev["prior"],
                       prev["noise"], prev["mismatch"], prev["seed"],
                       prev["method"]))
    rows = []
    t0 = time.perf_counter()
    for drift, spectrum, prior, noise, mismatch in MISMATCH_CONFIGS:
        for seed in range(5):
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum,
                drift=drift, noise=noise, mismatch=mismatch)
            G_init = make_prior(prior, x, y, w_true, t1, T_warm, Q0, lam0)
            y_te = y[T_warm:]
            n_te = len(y_te)

            # phases
            if drift in ("abrupt", "both"):
                regs = {"pre": slice(0, t1 - T_warm),
                        "dur": slice(t1 - T_warm, t1 - T_warm + 500),
                        "post": slice(t1 - T_warm + 500, n_te)}
            elif drift in ("gradual",):
                regs = {"pre": slice(0, t1 - T_warm),
                        "dur": slice(t1 - T_warm, t2 - T_warm),
                        "post": slice(t2 - T_warm, n_te)}
            else:
                regs = {"pre": slice(0, n_te),
                        "dur": slice(0, 0), "post": slice(0, 0)}

            preds_all = run_mismatch_arms(x, y, t1, t2, T_warm, G_init)
            for name, preds in preds_all.items():
                if (spectrum, drift, prior, noise, mismatch, seed,
                        name) in done:
                    continue
                row = {"spectrum": spectrum, "drift": drift, "prior": prior,
                       "noise": noise, "mismatch": mismatch, "seed": seed,
                       "method": name, "D": D, "T": T,
                       "wall_s": time.perf_counter() - t0}
                for reg_name, reg in regs.items():
                    row[f"mse_{reg_name}"] = float(
                        np.mean((preds[reg] - y_te[reg]) ** 2)) \
                        if reg.stop > reg.start else float("nan")
                row["mse_overall"] = float(np.mean((preds - y_te) ** 2))
                row["diverged"] = bool(row["mse_overall"] > 1e3
                                       or not np.all(np.isfinite(preds)))
                rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv(out_csv, mode="a",
                                          header=not out_csv.exists(),
                                          index=False)
                rows = []
            print(f"[{mismatch}/{drift}/{spectrum}] seed {seed} done "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.read_csv(out_csv)
    print(f"saved {out_csv} ({len(df)} rows)")

    # analysis (df holds the saved results; rows was flushed above)
    print("\n=== mismatched-generator results (mean over 5 seeds) ===")
    for mismatch, g in df.groupby("mismatch"):
        print(f"\n--- mismatch = {mismatch} ---")
        for name, g2 in g.groupby("method"):
            print(f"  {name:<20} overall={g2.mse_overall.mean():10.4f}  "
                  f"pre={g2[g2.mse_pre.notna()].mse_pre.mean():10.4f}  "
                  f"post={g2[g2.mse_post.notna()].mse_post.mean():10.4f}  "
                  f"diverged={int(g2.diverged.sum())}/{len(g2)}")

    # compare with the matched case (from the main grid)
    print("\n=== reference: matched case (from main grid, same configs) ===")
    try:
        frames = [pd.read_csv(RES / f"factorial_{t}.csv")
                  for t in ("d_abrupt", "d_covrot", "d_both", "extras")]
        df2 = pd.concat(frames, ignore_index=True)
        for name in ("spar_pull", "spar_eg", "kf_est", "leaky[0.997]"):
            sub = df2[(df2.method == name) & (df2.spectrum == "decay3")
                      & (df2.drift == "abrupt")]
            if len(sub):
                print(f"  {name:<20} overall={sub.mse_overall.mean():10.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
