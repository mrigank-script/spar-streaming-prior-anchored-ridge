"""Proposition 4 empirical scaling: OGD regret vs horizon T.

For T in {2000, 8000, 32000, 128000} on the three-phase stream (D=32), with
the drift phase held at a FIXED duration of 1000 steps at every horizon (so
the per-step drift rate is constant across T and does not dilute with the
horizon):
  - run OGD on the anchor pull s over the segment family,
  - run the fixed-s comparison grid,
  - regret(T) = total_loss(OGD) - min_s total_loss(fixed s),
  - per-step regret = regret(T)/T.
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

from ogd_spar_test import run_ogd_spar, run_segment_fixed  # noqa: E402
from spar_test import make_stream  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    T_warm = 1000
    horizons = [2000, 8000, 32000, 128000]
    seeds = 5
    rows = []
    for decay in (0.0, 3.0):
        for T in horizons:
            regrets = []
            per_step = []
            for seed in range(seeds):
                x, y, w_true, t1, t2, _, _ = make_stream(
                    seed=seed, D=32, T=T + T_warm, noise=0.1,
                    spectrum_decay=decay, drift_duration=1000)
                y_te = y[T_warm:]
                ogd, _ = run_ogd_spar(x, y, T_warm, alpha=0.997)
                loss_ogd = float(np.sum((ogd - y_te) ** 2))
                best = np.inf
                for sv in np.linspace(0.0, 1.0, 11):
                    p = run_segment_fixed(x, y, T_warm, float(sv), alpha=0.997)
                    l = float(np.sum((p - y_te) ** 2))
                    best = min(best, l)
                regrets.append(loss_ogd - best)
                per_step.append((loss_ogd - best) / len(y_te))
            rows.append({"spectrum_decay": decay, "T": T,
                         "regret_mean": float(np.mean(regrets)),
                         "regret_std": float(np.std(regrets)),
                         "per_step_regret": float(np.mean(per_step))})
            print(f"decay={decay} T={T:>7}: regret={np.mean(regrets):>9.2f} "
                  f"per-step={np.mean(per_step):.5f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "regret_scaling.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")

    # log-log slope of per-step regret vs T (theory: -1/2)
    for decay in (0.0, 3.0):
        sub = df[df.spectrum_decay == decay]
        slope = np.polyfit(np.log(sub["T"]), np.log(sub["per_step_regret"]), 1)[0]
        print(f"decay={decay}: log-log slope of per-step regret = {slope:.3f}"
              f"  (theory: -0.5)")

    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
