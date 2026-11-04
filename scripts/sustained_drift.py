"""Sustained-drift detector: fixed slow horizon vs the scale-free sticky floor.

The scope table records that a drift persisting beyond the slow horizon
`h_slow` relaxes both residual averages to a common level, so the level-
difference detector closes the window.  The sticky floor updates the slow
baseline only while no excess is present, so under sustained drift it stays
frozen at the pre-drift noise level and the excess remains positive.

Stream: a constant-velocity target after the warm-up (no jump), sustained
over a horizon much longer than h_slow = 1/base_eta = 20000.

Writes results/sustained_drift.csv.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rls_baselines import run_spar_kf  # noqa: E402

D, T, T_WARM = 128, 80000, 20000
NU = 1.5e-3
ARM = dict(prediction_units=True, debias_variance_floor=True,
           refresh_every=10, resid_fast=30, return_diag=True)


def sustained_stream(kind, seed=0, sigma=0.1):
    """Constant-velocity target after the warm-up: w_true(t) = w0 + nu*(t-T_WARM)*d."""
    rng = np.random.default_rng(seed)
    lam0 = np.ones(D) if kind == "iso" else np.logspace(0, -3, D)
    Q0, _ = np.linalg.qr(rng.normal(size=(D, D)))
    X = (rng.normal(size=(T, D)) * np.sqrt(lam0)) @ Q0.T
    w0 = rng.normal(scale=0.5, size=D)
    d = rng.normal(size=D)
    d /= np.linalg.norm(d)
    ramp = np.clip(np.arange(T) - T_WARM, 0, None)
    w_true = w0[None, :] + (NU * ramp)[:, None] * d[None, :]
    y = np.einsum("td,td->t", X, w_true) + sigma * rng.normal(size=T)
    return X, y, w0


def main() -> None:
    rows = []
    print(f"{'spec':<8} {'floor':<9} | {'mean w':>8} {'mean rho':>9} "
          f"{'tail MSE':>9} {'frac open':>9}")
    for kind in ("iso", "decay3"):
        x, y, w0 = sustained_stream(kind)
        G_init = w0 + 0.02 * np.random.default_rng(1).normal(size=D)
        y_te = y[T_WARM:]
        for sticky in (False, True):
            preds, rhos, ws = run_spar_kf(
                x, y, T_WARM, G_init=G_init, sticky_floor=sticky, **ARM)
            tail = slice(-30000, None)
            w_tail = float(np.mean(ws[tail]))
            rho_tail = float(np.mean(rhos[tail]))
            mse_tail = float(np.mean((preds[tail] - y_te[tail]) ** 2))
            frac_open = float(np.mean(ws[tail] > 1.5 * 5e-4))
            rows.append(dict(spectrum=kind, sticky_floor=sticky,
                             mean_w=w_tail, mean_rho=rho_tail,
                             tail_mse=mse_tail, frac_open=frac_open))
            name = "sticky" if sticky else "fixed"
            print(f"{kind:<8} {name:<9} | {w_tail:>8.5f} {rho_tail:>9.5f} "
                  f"{mse_tail:>9.5f} {frac_open:>9.2f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "sustained_drift.csv",
                              index=False)
    print("\nsaved results/sustained_drift.csv")


if __name__ == "__main__":
    main()
