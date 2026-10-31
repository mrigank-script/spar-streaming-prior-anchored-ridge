"""Ablation suite for SPAR-Pull: every component on/off + strength
sensitivity, on 4 representative configs x 5 seeds. Resumable by
(config, seed, variant). The full variant matches the shipped spar_kf2
arm (variance-floor-subtracted estimator); the no_vfloor variant removes
the floor subtraction."""

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

CONFIGS = [("abrupt", "decay3", "warm", "gauss"),
           ("covrot", "iso", "warm", "gauss"),
           ("abrupt", "iso", "adversarial", "gauss"),
           ("osc", "iso", "warm", "gauss")]

VARIANTS = {
    "full": dict(),
    "gamma=0.5": dict(gamma=0.5),
    "no_refresh": dict(refresh=False),
    "fixed_rho": dict(adaptive_rho=False),
    "fixed_w": dict(w_min=5e-4, w_max=5e-4),
    "no_cusum": dict(use_cusum=False),
    "no_ref": dict(use_ref=False),
    "cad50": dict(refresh_every=50),
    "wmax0.02": dict(w_max=0.02),
    "wmax0.10": dict(w_max=0.10),
    "rf100": dict(resid_fast=100),
    "etanu0.1": dict(eta_nu=0.1),
    "etanu0.4": dict(eta_nu=0.4),
    "no_vfloor": dict(debias_variance_floor=False),
}


def main() -> None:
    T_warm, D, T = 2000, 128, 10000
    out_csv = ROOT / "results" / "ablation_spar_kf.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["drift"], prev["spectrum"], prev["prior"],
                       prev["seed"], prev["variant"]))
    rows = []
    t0 = time.perf_counter()
    for drift, spectrum, prior, noise in CONFIGS:
        for seed in range(5):
            if all((drift, spectrum, prior, seed, v) in done
                   for v in VARIANTS):
                continue
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum,
                drift=drift, noise=noise)
            G_init = make_prior(prior, x, y, w_true, t1, T_warm, Q0, lam0)
            y_te = y[T_warm:]
            regs = phases_for(drift, t1 - T_warm, t2 - T_warm, len(y_te))
            for vname, kw in VARIANTS.items():
                if (drift, spectrum, prior, seed, vname) in done:
                    continue
                preds, _, _ = run_spar_kf(
                    x, y, T_warm, G_init=G_init,
                    refresh_every=kw.get("refresh_every", 10),
                    resid_fast=kw.get("resid_fast", 30),
                    rho_drift_floor=0.02, return_diag=True,
                    debias_variance_floor=kw.get("debias_variance_floor",
                                                 True),
                    **{k: v for k, v in kw.items()
                       if k not in ("refresh_every", "resid_fast",
                                    "debias_variance_floor")})
                row = {"drift": drift, "spectrum": spectrum, "prior": prior,
                       "seed": seed, "variant": vname,
                       "mse_overall": float(np.mean((preds - y_te) ** 2))}
                for reg_name, reg in regs.items():
                    if reg.stop > reg.start:
                        row[f"mse_{reg_name}"] = float(
                            np.mean((preds[reg] - y_te[reg]) ** 2))
                rows.append(row)
            print(f"[{drift}/{spectrum}/{prior}] seed {seed} done "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(out_csv, mode="a",
                                  header=not out_csv.exists(),
                                  index=False)
    df = pd.read_csv(out_csv)
    print(f"saved results/ablation_spar_kf.csv ({len(df)} rows)")

    print("\n=== Ablation: mean mse_overall across configs x seeds ===")
    g = df.groupby("variant")["mse_overall"].agg(["mean", "max"]).sort_values("mean")
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
