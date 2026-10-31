"""Dimension scaling of the Proposition 4(i) regret.

At a fixed horizon (T = 9000, drift phase a fixed 1,000 steps, as in the
horizon-scaling study), the per-step regret of the scalar segment arm and
the D-dimensional profile arm against the best fixed scalar pull
(11-point grid, each comparator arm with its own refresh trajectory),
across D in {8, 32, 128}:

  - the segment arm's guarantee is the scalar Zinkevich bound O(sqrt(T)),
    independent of D at fixed T;
  - the profile arm carries the sqrt(D) dimension factor of the
    Proposition 4(i) bound, so its per-step regret should grow ~sqrt(D).

Writes results/regret_dscaling.csv.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ogd_spar_test import run_ogd_profile, run_ogd_spar, run_segment_fixed  # noqa: E402
from spar_test import make_stream  # noqa: E402

RES = ROOT / "results"


def main() -> None:
    t0 = time.perf_counter()
    T_warm = 1000
    T = 9000
    horizons_drift = 1000
    dims = (8, 32, 128)
    decays = (0.0, 3.0)
    seeds = 5
    rows = []
    for D in dims:
        for decay in decays:
            for seed in range(seeds):
                x, y, w_true, t1, t2, _, _ = make_stream(
                    seed=seed, D=D, T=T, noise=0.1,
                    spectrum_decay=decay, drift_duration=horizons_drift)
                y_te = y[T_warm:]
                n_te = len(y_te)

                ogd_s, _ = run_ogd_spar(x, y, T_warm, alpha=0.997)
                ogd_d, _ = run_ogd_profile(x, y, T_warm, alpha=0.997)
                loss_s = float(np.sum((ogd_s - y_te) ** 2))
                loss_d = float(np.sum((ogd_d - y_te) ** 2))

                best = np.inf
                for sv in np.linspace(0.0, 1.0, 11):
                    p = run_segment_fixed(x, y, T_warm, float(sv), alpha=0.997)
                    l = float(np.sum((p - y_te) ** 2))
                    best = min(best, l)

                rows.append({
                    "D": D, "spectrum_decay": decay, "seed": seed,
                    "per_step_regret_segment": (loss_s - best) / n_te,
                    "per_step_regret_profile": (loss_d - best) / n_te,
                })
            print(f"[regret_dscaling] D={D} decay={decay} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "regret_dscaling.csv", index=False)

    print("\n=== per-step regret vs best fixed scalar pull (mean over seeds) ===")
    agg = df.groupby(["D", "spectrum_decay"]).mean(numeric_only=True)
    print(agg.round(6))


if __name__ == "__main__":
    main()
