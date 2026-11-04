"""Damped-trace variance estimator: replaces tr(A_bar^-1) with
tr((A_bar + rho*I)^-1) to prevent explosion on rank-deficient spectra.
Runs SPAR-Pull with both estimators on the rank-deficient configs only."""

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

from rls_baselines import make_prior, make_stream_fac, run_spar_kf
from synthetic_factorial import phases_for

RES = ROOT / "results"

RANKDEF = [("abrupt", "rankdef", p, "gauss") for p in ("oracle", "warm", "adversarial")] + \
          [("covrot", "rankdef", p, "gauss") for p in ("oracle", "warm", "adversarial")]


def run_spar_kf_damped(x, y, T_warm, G_init=None, refresh_every=10, resid_fast=30,
                       rho_drift_floor=0.02, ridge=0.01):
    """SPAR-Pull with the ridge-damped trace estimator."""
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    W_warm_fit = np.linalg.solve(A + n_eff * ridge * np.eye(D), B)
    G = np.asarray(G_init, dtype=np.float64).copy() if G_init is not None else W_warm_fit.copy()
    G_ref = G.copy()
    r_warm = y[:T_warm] - x[:T_warm] @ W_warm_fit
    sigma2_base = float(np.var(r_warm)) + 1e-12
    sigma2_fast = sigma2_base
    Abar_warm = A / n_eff
    rho_init = 0.1

    eta_f = 1.0 / resid_fast
    eta_ref = refresh_every / 2000.0
    C_floor_base = 0.5 * sigma2_base * D / (T_warm * max(np.trace(Abar_warm) / D, 1e-8))
    C_ema = 2 * C_floor_base
    nu2_ema = 0.0
    w = 5e-4
    alpha = 1.0 - w
    rho = rho_init
    preds = np.zeros(len(y) - T_warm)

    for t in range(T_warm, len(y)):
        if (t - T_warm) % refresh_every == 0:
            Abar = A / n_eff
            mu_bar = float(np.trace(Abar) / D) + 1e-12
            # DAMPED trace: use (Abar + rho*I)^-1 instead of Abar^-1
            rho_prev = max(rho, 1e-6)
            tr_inv_damped = float(np.trace(np.linalg.inv(Abar + rho_prev * np.eye(D))))
            V_hat = sigma2_base * tr_inv_damped / n_eff
            W_data = np.linalg.solve(A + n_eff * 1e-6 * np.eye(D), B)
            E_resid = max(0.0, sigma2_fast - sigma2_base)

            C_ref = float(np.sum((G_ref - W_data) ** 2)) - V_hat
            C_resid = E_resid * tr_inv_damped / (n_eff * mu_bar)
            cands = [C_ref, C_resid, C_floor_base, 1e-10]
            C_raw = max(cands)
            C_ema = (1 - 0.15) * C_ema + 0.15 * C_raw
            rho_raw = mu_bar * V_hat / C_ema
            lo = 1e-5
            if rho_drift_floor is not None and w >= 0.5 * 0.05:
                lo = max(lo, rho_drift_floor)
            rho = float(np.clip(rho_raw, lo, 1.0))

            s_bar = rho / (mu_bar + rho)
            c_prime = (1.0 - s_bar) ** 2 * sigma2_base * D
            nu2_hat = max(E_resid * w * w - c_prime * w ** 3 / max(2.0 - w, 1e-6), 0.0)
            nu2_ema = (1 - 0.2) * nu2_ema + 0.2 * nu2_hat
            kappa = 4.0 * nu2_ema / max(c_prime, 1e-12)
            w_star = kappa ** (1.0 / 3.0)
            w = float(np.clip(w_star, 5e-4, 0.05))
            alpha = 1.0 - w

        A_eff = A + n_eff * rho * np.eye(D)
        B_eff = B + n_eff * rho * G
        W = np.linalg.solve(A_eff, B_eff)
        preds[t - T_warm] = x[t] @ W

        r_t = y[t] - x[t] @ W
        sigma2_fast = (1 - eta_f) * sigma2_fast + eta_f * (r_t * r_t)
        sigma2_base = (1 - 5e-5) * sigma2_base + 5e-5 * (r_t * r_t)

        if (t - T_warm) % refresh_every == 0 and t > T_warm:
            Abar = A / n_eff
            mu, U = np.linalg.eigh(Abar)
            beta = mu / (mu + rho)
            Gp = U.T @ G
            Wp = U.T @ W
            G = U @ (beta * Wp + (1 - beta) * Gp)
            G_ref = (1 - eta_ref) * G_ref + eta_ref * W

        A = alpha * A + np.outer(x[t], x[t])
        B = alpha * B + x[t] * y[t]
        n_eff = alpha * n_eff + 1.0

    return preds


def main():
    T_warm, D, T = 2000, 128, 10000
    out_csv = RES / "damped_trace.csv"
    rows = []
    t0 = time.perf_counter()
    for drift, spectrum, prior, noise in RANKDEF:
        for seed in range(5):
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum,
                drift=drift, noise=noise)
            G_init = make_prior(prior, x, y, w_true, t1, T_warm, Q0, lam0)
            y_te = y[T_warm:]
            regs = phases_for(drift, t1 - T_warm, t2 - T_warm, len(y_te))
            for name, fn in [
                ("spar_kf2", lambda: run_spar_kf(
                    x, y, T_warm, G_init=G_init, refresh_every=10, resid_fast=30,
                    rho_drift_floor=0.02, debias_variance_floor=True)[0]),
                ("spar_kf_damped", lambda: run_spar_kf_damped(
                    x, y, T_warm, G_init=G_init, refresh_every=10, resid_fast=30,
                    rho_drift_floor=0.02)),
            ]:
                preds = fn()
                row = {"spectrum": spectrum, "drift": drift, "prior": prior,
                       "seed": seed, "method": name, "D": D, "T": T,
                       "wall_s": time.perf_counter() - t0}
                for reg_name, reg in regs.items():
                    row[f"mse_{reg_name}"] = float(
                        np.mean((preds[reg] - y_te[reg]) ** 2)) \
                        if reg.stop > reg.start else float("nan")
                row["mse_overall"] = float(np.mean((preds - y_te) ** 2))
                row["diverged"] = bool(row["mse_overall"] > 1e3
                                       or not np.all(np.isfinite(preds)))
                rows.append(row)
            print(f"[damped] {drift}/{spectrum}/{prior} s{seed} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv} ({len(df)} rows)")
    print(df.groupby('method')['mse_overall'].agg(['mean', 'max']).round(4).to_string())


if __name__ == "__main__":
    main()
