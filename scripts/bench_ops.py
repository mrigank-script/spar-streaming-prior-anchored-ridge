"""Per-step cost table: wall-clock microseconds per streamed step for every
method at D in {128, 512, 1024, 2048}.

The warm-up is differenced out (two stream lengths), so the reported figure
is the marginal streaming cost.  ``spar_scalar(solve)`` forms an explicit
D x D solve every step (O(D^3)); ``spar_diag`` is O(D^2)/step with O(D^3)
only at the refresh cadence; ``spar_pull`` shares the O(D^3) solve family.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rls_baselines import (make_stream_fac, run_adam, run_arow, run_df,  # noqa: E402
                           run_kf, run_leaky_batch, run_rls, run_spar_diag,
                           run_spar_kf, run_sgd)
from spar_test import run_spar_v2  # noqa: E402


def _time(fn, x, y, T_warm, n_check=3):
    best = np.inf
    for _ in range(n_check):
        t0 = time.perf_counter()
        fn(x, y, T_warm)
        best = min(best, time.perf_counter() - t0)
    return best


def time_method(fn, x, y, T_warm, x2, y2, n_check=1):
    """Marginal streaming cost: difference of two stream lengths so the
    warm-up fit cancels; returns microseconds per step."""
    dt = _time(fn, x2, y2, T_warm, n_check) - _time(fn, x, y, T_warm, n_check)
    return dt / max(len(y2) - len(y), 1) * 1e6


def main() -> None:
    rows = []
    for D in (128, 512, 1024, 2048):
        T, T_warm = 800, 200
        x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
            seed=0, D=D, T=T, spectrum="decay3", drift="none")
        x2, y2, *_ = make_stream_fac(
            seed=0, D=D, T=T + 500, spectrum="decay3", drift="none")
        arms = {
            "leaky_rls": lambda x, y, w: run_rls(x, y, w, lam=0.997),
            "df": lambda x, y, w: run_df(x, y, w, lam_f=0.95),
            "arow": lambda x, y, w: run_arow(x, y, w, r=0.1),
            "kalman": lambda x, y, w: run_kf(x, y, w, q=1e-5),
            "sgd": lambda x, y, w: run_sgd(x, y, w, lr=0.003),
            "adam": lambda x, y, w: run_adam(x, y, w, lr=0.003),
            "spar_scalar(solve)": lambda x, y, w: run_spar_v2(
                x, y, w, alpha=0.997, refresh_every=50),
            "spar_diag(O2)": lambda x, y, w: run_spar_diag(
                x, y, w, alpha=0.997, refresh_every=50),
            "spar_pull": lambda x, y, w: run_spar_kf(
                x, y, w, G_init=None, refresh_every=10, resid_fast=30,
                rho_drift_floor=0.02, debias_variance_floor=True,
                prediction_units=True)[0],
            "leaky_batch": lambda x, y, w: run_leaky_batch(
                x, y, w, lam=0.99),
        }
        for name, fn in arms.items():
            us = time_method(fn, x, y, T_warm, x2, y2)
            rows.append({"D": D, "method": name, "us_per_step": us})
            print(f"D={D:>5} {name:<20} {us:>9.1f} us/step", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "ops_table.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")

    pivot = df.pivot(index="D", columns="method", values="us_per_step").round(1)
    print("\n", pivot.to_string())


if __name__ == "__main__":
    main()
