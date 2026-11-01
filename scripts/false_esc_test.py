"""Proposition 3 empirical validation: false-escalation rate of the CUSUM signal.

Runs N stationary streams (no drift anywhere) through the SPAR machinery
with a fixed, well-calibrated model (adaptive_rho=False, rho=0.1), records
the two-scale residual CUSUM statistic

    z_t = (sigma2_fast - sigma2_slow) / sigma2_true

and measures the empirical rate at which max_t z_t exceeds the analytic
windowed-Hoeffding threshold tau(delta) = sigma2 * sqrt(2 log(T/delta) / m)
with m = resid_fast (the fast EMA horizon). Proposition 3 predicts the false
rate is <= delta. We report empirical vs nominal for several delta.
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

from rls_baselines import make_stream_fac  # noqa: E402
from spar_test import run_spar_v2  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    N = 300
    D, T, T_warm = 64, 5000, 1500
    sigma2_true = 0.01
    m_eff = 100  # resid_fast horizon

    max_z = {s: [] for s in ("iso", "decay3")}
    for spectrum in ("iso", "decay3"):
        for seed in range(N):
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=1000 + seed, D=D, T=T, sigma=np.sqrt(sigma2_true),
                spectrum=spectrum, drift="none")
            _, _, (sf, ss) = run_spar_v2(
                x, y, T_warm, rho_init=0.1, alpha=1.0,
                adaptive_rho=False, refresh=True, return_diag=True)
            z = (sf - ss) / sigma2_true
            max_z[spectrum].append(float(np.max(z)))

    print(f"N={N} stationary streams per spectrum, D={D}, T={T}")
    print(f"{'delta':>8} {'nominal tau/sgm2':>18} {'empirical rate':>16}"
          f" {'<= nominal?':>12}")
    for delta in (0.10, 0.05, 0.01):
        tau = np.sqrt(2.0 * np.log(T / delta) / m_eff)
        row = f"{delta:>8.2f} {tau:>18.3f}"
        ok_all = True
        rates = []
        for spectrum in ("iso", "decay3"):
            mz = np.array(max_z[spectrum])
            rate = float(np.mean(mz > tau))
            rates.append(rate)
            ok_all &= (rate <= delta)
            row += f"   {spectrum}={rate:.3f}"
        row += f"   {'YES' if ok_all else 'NO':>4}"
        print(row)

    for spectrum in ("iso", "decay3"):
        mz = np.array(max_z[spectrum])
        print(f"max-z quantiles [{spectrum}]: "
              f"50%={np.quantile(mz, 0.5):.2f} "
              f"90%={np.quantile(mz, 0.9):.2f} "
              f"99%={np.quantile(mz, 0.99):.2f} "
              f"max={mz.max():.2f}")

    import pandas as pd
    pd.DataFrame({"max_z_iso": max_z["iso"], "max_z_decay3": max_z["decay3"]}
                 ).to_csv(ROOT / "results" / "false_esc.csv", index=False)
    print(f"saved results/false_esc.csv ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
