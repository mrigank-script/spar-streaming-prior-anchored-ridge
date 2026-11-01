"""Pilot: legacy parameter-unit vs prediction-unit conflict branch."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rls_baselines import make_stream_fac, make_prior, run_spar_kf  # noqa: E402


def main() -> None:
    rows = []
    t0 = time.perf_counter()
    for spectrum in ("iso", "decay3", "rankdef"):
        for drift in ("abrupt", "covrot", "both"):
            for seed in (0, 1):
                x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                    seed=seed, D=128, T=10000, sigma=0.1,
                    spectrum=spectrum, drift=drift)
                G = make_prior("warm", x, y, w_true, t1, 2000, Q0, lam0)
                out = {}
                for tag, pu in (("legacy", False), ("pred", True)):
                    preds, rhos = run_spar_kf(
                        x, y, 2000, G_init=G, refresh_every=10, resid_fast=30,
                        rho_drift_floor=0.02, debias_variance_floor=True,
                        prediction_units=pu)
                    mse = float(np.mean((preds - y[2000:]) ** 2))
                    out[tag] = (mse, float(np.mean(rhos)), float(np.mean(rhos[-1000:])))
                rows.append((spectrum, drift, seed, *out["legacy"], *out["pred"]))
                print(f"{spectrum:8s} {drift:7s} s{seed} "
                      f"legacy={out['legacy'][0]:.4f} rho={out['legacy'][1]:.4f} "
                      f"pred={out['pred'][0]:.4f} rho={out['pred'][1]:.4f} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    a = np.array([[r[3], r[6]] for r in rows])
    print("\nmean MSE legacy/pred:", a.mean(axis=0))
    print("median ratio pred/legacy:", np.median(a[:, 1] / a[:, 0]))
    print("pred wins:", int(np.sum(a[:, 1] < a[:, 0])), "/", len(a))


if __name__ == "__main__":
    main()
