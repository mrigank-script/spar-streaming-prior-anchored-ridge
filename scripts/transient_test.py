"""Transient post-jump test for the conflict estimator.

Compares the adaptive strength rule against fixed strengths on the window
immediately after an abrupt target jump, before the k-step anchor refresh
can overwrite the prior.  Prediction-unit SPAR (the paper's arm).
"""

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

from rls_baselines import make_prior, make_stream_fac, run_spar_kf  # noqa: E402

D, T, T_WARM = 128, 10000, 2000
PINS = (0.01, 0.1, 1.0)
CADENCES = (10, 50)
SEEDS = range(5)


def main() -> None:
    rows = []
    t0 = time.perf_counter()
    for spectrum in ("decay3", "iso"):
        for seed in SEEDS:
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum,
                drift="abrupt")
            G = make_prior("warm", x, y, w_true, t1, T_WARM, Q0, lam0)
            y_te = y[T_WARM:]
            t1p = int(t1 - T_WARM)
            for cadence in CADENCES:
                arms = {"adaptive": None}
                for pin in PINS:
                    arms[f"rho={pin:g}"] = pin
                for name, pin in arms.items():
                    preds = run_spar_kf(
                        x, y, T_WARM, G_init=G, refresh_every=cadence,
                        resid_fast=30, rho_drift_floor=0.02,
                        debias_variance_floor=True, prediction_units=True,
                        rho_pin=pin)[0]
                    err = (preds - y_te) ** 2
                    pre = err[:t1p]
                    post = err[t1p:t1p + 50]
                    rows.append({
                        "spectrum": spectrum, "seed": seed,
                        "cadence": cadence, "method": name,
                        "mse_pre": float(np.mean(pre)),
                        "mse_post50": float(np.mean(post)),
                        "mse_overall": float(np.mean(err)),
                    })
            print(f"[transient] {spectrum} s{seed} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "transient_test.csv"
    df.to_csv(out, index=False)
    print()
    print(df.groupby(["cadence", "method"])[
        ["mse_pre", "mse_post50", "mse_overall"]].mean().round(4).to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()