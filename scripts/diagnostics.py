"""Consolidated numerical diagnostics: one entry point for every
paper-cited diagnostic.

  py scripts/diagnostics.py --diag vff       # VFF saturation
  py scripts/diagnostics.py --diag windup     # symmetrization control
  py scripts/diagnostics.py --diag sgd        # SGD stability cliff
  py scripts/diagnostics.py --diag coupling   # closed-loop coupling
  py scripts/diagnostics.py --diag rmt        # spectral risk (adjudication,
                                              #   operating points, sensitivities)
  py scripts/diagnostics.py --diag subopt      # finite-cadence window surrogate gap
  py scripts/diagnostics.py --diag all

Replaces the six per-diagnostic scripts (vff_diag.py, windup_mech.py,
sgd_lr_cliff.py, coupling_diag.py, rmt_adjudicate.py, rmt_verify.py);
each body is verbatim. Every diagnostic writes the same CSV as the
script it replaces, so committed results are reproduced unchanged.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rls_baselines import (make_prior, make_stream_fac,  # noqa: E402
                           run_leaky_batch, run_sgd, run_spar_kf)

RES = ROOT / "results"


# ============================================================ VFF saturation
def vff_info_traced(x, y, T_warm, lam_min=0.99, lam0=1.0,
                    ridge_frac=0.01, recursive_xi=False, eta_xi=0.01):
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    r_warm = y[:T_warm] - x[:T_warm] @ np.linalg.solve(
        A + n_eff * ridge_frac * np.eye(D), B)
    sigma0 = float(np.var(r_warm)) + 1e-12
    xi = lam0 * sigma0
    alpha = 1.0
    preds = np.zeros(len(y) - T_warm)
    lams_t = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        phi = x[t]
        pred = phi @ np.linalg.solve(
            A + n_eff * ridge_frac * np.eye(D), B)
        preds[t - T_warm] = pred
        e = y[t] - pred
        lam = float(np.clip(1.0 - (e * e) / xi, lam_min, 1.0))
        lams_t[t - T_warm] = lam
        if recursive_xi:
            xi = (1.0 - eta_xi) * xi + eta_xi * (e * e)
        alpha = 1.0 - (1.0 - lam)
        A = alpha * A + np.outer(phi, phi)
        B = alpha * B + phi * y[t]
        n_eff = alpha * n_eff + 1.0
    return preds, lams_t


def diag_vff() -> None:
    T_warm, D, T = 2000, 128, 10000
    rows = []
    for drift, spectrum in (("abrupt", "decay3"), ("none", "iso")):
        for seed in range(3):
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum,
                drift=drift, noise="gauss")
            y_te = y[T_warm:]
            arms = {
                "vff_info[lmin=0.99]":
                    (vff_info_traced(x, y, T_warm, lam_min=0.99), 0.99),
                "vff_info[lmin=0.95]":
                    (vff_info_traced(x, y, T_warm, lam_min=0.95), 0.95),
                "vff_info[rec-xi,lmin=0.99]":
                    (vff_info_traced(x, y, T_warm, lam_min=0.99,
                                     recursive_xi=True), 0.99),
                "leaky_batch[0.99]":
                    ((run_leaky_batch(x, y, T_warm, lam=0.99), None), None),
                "leaky_batch[0.95]":
                    ((run_leaky_batch(x, y, T_warm, lam=0.95), None), None),
            }
            for name, (res, floor) in arms.items():
                preds, lams_t = res[0], res[1]
                row = {"drift": drift, "spectrum": spectrum, "seed": seed,
                       "method": name,
                       "mse_overall": float(np.mean((preds - y_te) ** 2))}
                if lams_t is not None:
                    row["frac_at_floor"] = float(
                        (lams_t <= floor + 1e-12).mean())
                    row["mean_lam"] = float(lams_t.mean())
                    row["mean_lam_post_t1"] = float(
                        lams_t[t1 - T_warm:].mean())
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "vff_diag.csv", index=False)
    print(f"saved results/vff_diag.csv ({len(df)} rows)")
    print(df.groupby(["drift", "method"]).mean(numeric_only=True)
          .round(4).to_string())


# ==================================================== windup symmetrization
def _run_rls_variant(x, y, T_warm, lam, rho=0.1, symmetrize=False):
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + rho * T_warm * np.eye(D), b_w)
    P = np.linalg.inv(A_w / T_warm + rho * np.eye(D)) / T_warm
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        preds[t - T_warm] = x[t] @ W
        k = P @ x[t]
        gain = k / (lam + x[t] @ k)
        W = W + gain * (y[t] - x[t] @ W)
        P = (P - np.outer(gain, k)) / lam
        if symmetrize:
            P = 0.5 * (P + P.T)
    return preds


def diag_windup() -> None:
    T_warm = 2000
    rows = []
    for D, lam in ((128, 0.997), (128, 0.995), (128, 0.99), (512, 0.99)):
        for sym in (False, True):
            mses = []
            for drift, spectrum in (("abrupt", "decay3"), ("abrupt", "iso")):
                for seed in (0, 1):
                    x, y, *_ = make_stream_fac(
                        seed=seed, D=D, T=10000, sigma=0.1,
                        spectrum=spectrum, drift=drift, noise="gauss")
                    preds = _run_rls_variant(x, y, T_warm, lam,
                                             symmetrize=sym)
                    y_te = y[T_warm:]
                    mses.append(float(np.mean((preds - y_te) ** 2)))
            rows.append({
                "lam": lam, "n_eff": round(1.0 / (1.0 - lam)),
                "D": D, "symmetrized": sym,
                "mse_mean": float(np.mean(mses)),
                "mse_max": float(np.max(mses)),
                "n_runs": len(mses),
            })
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "windup_mech.csv", index=False)
    print("saved results/windup_mech.csv")


# ======================================================= SGD stability cliff
SGD_LRS = (0.003, 0.01, 0.02, 0.05, 0.1)
SGD_MAIN = [(d, s, p)
            for d in ("abrupt", "covrot", "both")
            for s in ("iso", "decay3", "rankdef")
            for p in ("oracle", "warm", "adversarial")]


def diag_sgd() -> None:
    D, T, T_warm, sigma, seeds = 128, 10000, 2000, 0.1, 5
    rows = []
    t0 = time.perf_counter()
    for drift, spectrum, prior in SGD_MAIN:
        for seed in range(seeds):
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=D, T=T, sigma=sigma, spectrum=spectrum,
                drift=drift, noise="gauss")
            make_prior(prior, x, y, w_true, t1, T_warm, Q0, lam0)
            y_te = y[T_warm:]
            for lr in SGD_LRS:
                preds = run_sgd(x, y, T_warm, lr=lr)
                rows.append({
                    "drift": drift, "spectrum": spectrum, "prior": prior,
                    "seed": seed, "method": f"sgd[{lr}]",
                    "mse_overall": float(np.mean((preds - y_te) ** 2)),
                    "diverged": bool(np.mean((preds - y_te) ** 2) > 1e3
                                     or not np.all(np.isfinite(preds))),
                })
        print(f"[{drift}/{spectrum}/{prior}] {time.perf_counter()-t0:.0f}s",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sgd_lr_cliff.csv", index=False)

    print("\n=== SGD lr sweep: mean / worst MSE over 27 main configs ===")
    for lr in SGD_LRS:
        sub = df[df.method == f"sgd[{lr}]"]
        ndiv = int(sub.diverged.sum())
        print(f"sgd[{lr}]: mean={sub.mse_overall.mean():.4f}  "
              f"worst={sub.mse_overall.max():.4f}  diverged={ndiv}/135")


# ==================================================== closed-loop coupling
COUPLING_CONFIGS = [("abrupt", "decay3", "warm", "gauss"),
                    ("covrot", "iso", "warm", "gauss"),
                    ("abrupt", "iso", "adversarial", "gauss"),
                    ("osc", "iso", "warm", "gauss")]
COUPLING_ARM = dict(refresh_every=10, resid_fast=30, rho_drift_floor=0.02,
                    debias_variance_floor=True)
RHO_PIN_CLAMP = (1e-4, 10.0)
COUPLING_TAIL = 1000
COUPLING_OUT = RES / "coupling_diag.csv"
COUPLING_COLS = ["drift", "spectrum", "prior", "seed",
                 "rho_base", "u_base", "mse_base",
                 "L_urho", "L_rhou", "product", "condition_holds",
                 "rho_pin_lo", "rho_pin_hi", "u_star_lo", "u_star_hi",
                 "w_pin_lo", "w_pin_hi", "rho_star_lo", "rho_star_hi",
                 "mse_rho_lo", "mse_rho_hi", "mse_w_lo", "mse_w_hi",
                 "n_diverged_pinned"]


def _converged(diag, tail=COUPLING_TAIL):
    rhos, ws = diag[1], diag[2]
    return (float(np.mean(rhos[-tail:])),
            float(np.mean(ws[-tail:] ** 3)),
            float(np.mean(ws[-tail:])))


def _run_pin(x, y, T_warm, G_init, y_te, **pin):
    preds, rhos, ws = run_spar_kf(x, y, T_warm, G_init=G_init,
                                  return_diag=True, **COUPLING_ARM, **pin)
    rho_bar, u_bar, _ = _converged((preds, rhos, ws))
    return (rho_bar, u_bar, float(np.mean((preds - y_te) ** 2)), preds)


def diag_coupling() -> None:
    T_warm, D, T = 2000, 128, 10000
    done = set()
    if COUPLING_OUT.exists():
        prev = pd.read_csv(COUPLING_OUT)
        done = set(zip(prev["drift"], prev["spectrum"], prev["prior"],
                       prev["seed"]))
    t0 = time.perf_counter()
    for drift, spectrum, prior, noise in COUPLING_CONFIGS:
        for seed in range(5):
            if (drift, spectrum, prior, seed) in done:
                continue
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum,
                drift=drift, noise=noise)
            G_init = make_prior(prior, x, y, w_true, t1, T_warm, Q0, lam0)
            y_te = y[T_warm:]

            preds, rhos, ws = run_spar_kf(x, y, T_warm, G_init=G_init,
                                          return_diag=True, **COUPLING_ARM)
            rho_base, u_base, _ = _converged((preds, rhos, ws))
            mse_base = float(np.mean((preds - y_te) ** 2))

            rho_pin_lo = float(np.clip(0.9 * rho_base, *RHO_PIN_CLAMP))
            rho_pin_hi = float(np.clip(1.1 * rho_base, *RHO_PIN_CLAMP))
            _, u_star_lo, mse_rho_lo, pr_lo = _run_pin(
                x, y, T_warm, G_init, y_te, rho_pin=rho_pin_lo)
            _, u_star_hi, mse_rho_hi, pr_hi = _run_pin(
                x, y, T_warm, G_init, y_te, rho_pin=rho_pin_hi)
            d_rho = abs(rho_pin_hi - rho_pin_lo)
            L_urho = (abs(u_star_hi - u_star_lo) / d_rho
                      if d_rho > 0 else float("nan"))

            u_lo, u_hi = 0.9 * u_base, 1.1 * u_base
            w_pin_lo, w_pin_hi = u_lo ** (1.0 / 3.0), u_hi ** (1.0 / 3.0)
            rho_star_lo, _, mse_w_lo, pw_lo = _run_pin(
                x, y, T_warm, G_init, y_te, w_pin=w_pin_lo)
            rho_star_hi, _, mse_w_hi, pw_hi = _run_pin(
                x, y, T_warm, G_init, y_te, w_pin=w_pin_hi)
            d_u = abs(u_hi - u_lo)
            L_rhou = (abs(rho_star_hi - rho_star_lo) / d_u
                      if d_u > 0 else float("nan"))

            product = L_urho * L_rhou
            condition_holds = bool(product < 1) if np.isfinite(product) \
                else False
            n_div = sum(float(np.mean((p - y_te) ** 2)) > 1e3
                        or not np.all(np.isfinite(p))
                        for p in (pr_lo, pr_hi, pw_lo, pw_hi))

            row = dict(zip(COUPLING_COLS, [
                drift, spectrum, prior, seed,
                rho_base, u_base, mse_base,
                L_urho, L_rhou, product, condition_holds,
                rho_pin_lo, rho_pin_hi, u_star_lo, u_star_hi,
                w_pin_lo, w_pin_hi, rho_star_lo, rho_star_hi,
                mse_rho_lo, mse_rho_hi, mse_w_lo, mse_w_hi,
                n_div]))
            pd.DataFrame([row]).to_csv(COUPLING_OUT, mode="a",
                                        header=not COUPLING_OUT.exists(),
                                        index=False)
            print(f"[coupling] {drift}/{spectrum}/{prior} s{seed} "
                  f"P={product:.6g} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.read_csv(COUPLING_OUT)
    g = df.groupby(["drift", "spectrum", "prior"]).mean(numeric_only=True)
    print(g[["rho_base", "u_base", "L_urho", "L_rhou", "product"]]
          .round(6).to_string())
    print(f"condition holds: {int(df.condition_holds.sum())}"
          f"/{len(df)} runs")


# ============================================================ spectral risk
RMT_D = 128
RMT_SIGMA2 = 0.01
RMT_T = 10000
RMT_T_WARM = 2000


def rmt_spectrum(kind):
    if kind == "iso":
        return np.ones(RMT_D)
    if kind == "decay3":
        return np.logspace(0, -3, RMT_D)
    if kind == "rankdef":
        return np.concatenate([np.logspace(0, -2, RMT_D // 2),
                               np.zeros(RMT_D // 2)])
    raise ValueError(kind)


def rmt_solve_q(lam, w, rho):
    c = RMT_D * w
    lo, hi = 1e-12, 1.0
    for _ in range(200):
        q = 0.5 * (lo + hi)
        x = c * q
        grow = np.expm1(x) / x if x > 0 else 1.0
        if np.mean(lam / (lam + rho * grow)) - q > 0:
            lo = q
        else:
            hi = q
    q = 0.5 * (lo + hi)
    x = c * q
    grow = np.expm1(x) / x if x > 0 else 1.0
    return q, rho * grow


def rmt_formulas(lam, w, rho):
    q, rho_eff = rmt_solve_q(lam, w, rho)
    c = RMT_D * w
    x = c * q
    Jt = float(np.mean((lam / (lam + rho_eff)) ** 2))
    Gam = (np.expm1(x) - x) / x**2 if x > 1e-12 else 0.5
    amp = (x + np.exp(-x) - 1) / x**2 if x > 1e-12 else 0.5
    v_a = RMT_SIGMA2 * c * Gam * Jt / (1 - c * amp * Jt)
    Ups = (Jt / q) * (1 - (1 - np.exp(-x)) / x) if x > 1e-12 else 0.0
    v_b = RMT_SIGMA2 * Ups / (1 - Ups)
    v_c = (RMT_SIGMA2 * c / (1 - c) if c < 1 else float("inf"))
    return q, rho_eff, v_a, v_b, v_c


def rmt_mc_fixed(kind, w, rho, n_streams=40):
    """Direct simulation at fixed (w, rho): stationary stream, geometric
    accumulator, oracle anchor, fresh-test excess risk."""
    excesses = []
    for seed in range(n_streams):
        x, y, w_true, *_ = make_stream_fac(
            seed=seed, D=RMT_D, T=RMT_T, sigma=np.sqrt(RMT_SIGMA2),
            spectrum=kind, drift="none", noise="gauss")
        w_star = w_true[-1]
        sw = w * (1 - w) ** np.arange(RMT_T)
        Xw = x * np.sqrt(sw)[:, None]
        A = Xw.T @ Xw
        B = Xw.T @ (y * np.sqrt(sw))
        W = np.linalg.solve(A + rho * np.eye(RMT_D), B + rho * w_star)
        Sig_hat = np.cov(x, rowvar=False)
        dW = W - w_star
        excesses.append(float(dW @ Sig_hat @ dW))
    return float(np.mean(excesses))


def rmt_de_mse(lam, w, rho):
    c = RMT_D * w

    def q_of(rho_):
        lo, hi = 1e-12, 1.0
        for _ in range(200):
            q = 0.5 * (lo + hi)
            x = c * q
            grow = np.expm1(x) / x if x > 0 else 1.0
            if np.mean(lam / (lam + rho_ * grow)) - q > 0:
                lo = q
            else:
                hi = q
        return 0.5 * (lo + hi)

    q = q_of(rho)
    x = c * q
    rho_eff = rho * (np.expm1(x) / x if x > 0 else 1.0)
    Jt = float(np.mean((lam / (lam + rho_eff)) ** 2))
    Ups = (Jt / q) * (1 - (1 - np.exp(-x)) / x) if x > 1e-12 else 0.0
    return q, rho_eff, RMT_SIGMA2 * (1 + Ups / (1 - Ups)), Jt


def rmt_sens_check(lam, w, rho):
    c = RMT_D * w
    eps = 1e-6

    def sbar(rho_):
        lo, hi = 1e-12, 1.0
        for _ in range(200):
            q = 0.5 * (lo + hi)
            x = c * q
            grow = np.expm1(x) / x if x > 0 else 1.0
            if np.mean(lam / (lam + rho_ * grow)) - q > 0:
                lo = q
            else:
                hi = q
        return 1 - 0.5 * (lo + hi)

    num = (sbar(rho * (1 + eps)) - sbar(rho * (1 - eps))) / (2 * eps)
    q, rho_eff, _, _ = rmt_de_mse(lam, w, rho)
    x = c * q
    s_t = rho_eff / (lam + rho_eff)
    I_H = float(np.mean(s_t * (1 - s_t)))
    phi = 1 / (1 - np.exp(-x)) - 1 / x if x > 1e-12 else 0.5
    closed = I_H / (1 + c * phi * I_H)
    return num, closed


# ------------------------------------------- Theorem 4(iii): conflict bias
def rmt_conflict_formula(lam, w, rho, dcoef):
    """Theorem 4(iii): theta_eff, Upsilon, oracle variance, conflict bias.

    H_delta carries an atom of mass dcoef_i^2 at direction tau_i
    (total ||delta||^2); bias = rho_eff^2 * int tau/(tau+rho_eff)^2 dH_delta
    / (1 - Upsilon)."""
    q, rho_eff = rmt_solve_q(lam, w, rho)
    c = RMT_D * w
    x = c * q
    Jt = float(np.mean((lam / (lam + rho_eff)) ** 2))
    Ups = (Jt / q) * (1 - (1 - np.exp(-x)) / x) if x > 1e-12 else 0.0
    var_oracle = RMT_SIGMA2 * Ups / (1 - Ups)
    bias = (rho_eff ** 2
            * float(np.sum(dcoef ** 2 * lam / (lam + rho_eff) ** 2))
            / (1 - Ups))
    return q, rho_eff, Ups, var_oracle, bias


def rmt_conflict_point(kind, w, rho, dcoef, n_streams=40):
    """Paired MC totals: conflict anchor (G = w* + Q0 d) vs oracle anchor."""
    sw = w * (1 - w) ** np.arange(RMT_T)
    eye = np.eye(RMT_D)
    tot_d, tot_o = [], []
    for seed in range(n_streams):
        x, y, w_true, _t1, _t2, Q0, _lam0 = make_stream_fac(
            seed=seed, D=RMT_D, T=RMT_T, sigma=np.sqrt(RMT_SIGMA2),
            spectrum=kind, drift="none", noise="gauss")
        w_star = w_true[-1]
        Xw = x * np.sqrt(sw)[:, None]
        A = Xw.T @ Xw
        B = Xw.T @ (y * np.sqrt(sw))
        delta = Q0 @ dcoef
        Wd = np.linalg.solve(A + rho * eye, B + rho * (w_star + delta))
        Wo = np.linalg.solve(A + rho * eye, B + rho * w_star)
        Sig = np.cov(x, rowvar=False)
        tot_d.append(float((Wd - w_star) @ Sig @ (Wd - w_star)))
        tot_o.append(float((Wo - w_star) @ Sig @ (Wo - w_star)))
    return float(np.mean(tot_d)), float(np.mean(tot_o))


def diag_rmt_conflict() -> None:
    rng = np.random.default_rng(0)
    d_random = rng.normal(size=RMT_D)
    d_random /= np.linalg.norm(d_random)      # ||delta||^2 = 1
    d_top = np.zeros(RMT_D)
    d_top[0] = 1.0                            # all conflict on tau_max
    print("=== Theorem 4(iii): anchor-conflict bias (paired MC vs formula) ===")
    print(f"{'spec':<8} {'delta':<8} {'n_eff':>6} {'rho':>7} | {'rho_eff':>9} "
          f"| {'MC inc':>10} {'formula':>10} {'rel err':>8}")
    rows = []
    for kind in ("iso", "decay3", "rankdef"):
        lam = rmt_spectrum(kind)
        for dname, dcoef in (("random", d_random), ("top_dir", d_top)):
            for n_eff in (1000, 200):
                w = 1.0 / n_eff
                for rho in (1e-2, 1e-1):
                    q, rho_eff, Ups, var_o, bias_f = rmt_conflict_formula(
                        lam, w, rho, dcoef)
                    tot_d, tot_o = rmt_conflict_point(kind, w, rho, dcoef)
                    inc_mc = tot_d - tot_o
                    rel = abs(inc_mc - bias_f) / max(bias_f, 1e-12)
                    rows.append(dict(spectrum=kind, delta=dname,
                                     n_eff=n_eff, rho=rho, q=q,
                                     rho_eff=rho_eff, Ups=Ups,
                                     mc_conflict_total=tot_d, mc_oracle=tot_o,
                                     mc_increment=inc_mc,
                                     formula_bias=bias_f, rel_err=rel))
                    print(f"{kind:<8} {dname:<8} {n_eff:>6} {rho:>7} | "
                          f"{rho_eff:>9.4g} | {inc_mc:>10.5f} "
                          f"{bias_f:>10.5f} {rel:>8.1%}")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "rmt_conflict.csv", index=False)
    for dname in ("random", "top_dir"):
        sub = df[df.delta == dname]
        print(f"{dname}: median rel err {sub.rel_err.median():.1%}  "
              f"max {sub.rel_err.max():.1%}")
    print("saved results/rmt_conflict.csv")
    print()


def diag_rmt() -> None:
    print("=== adjudication: fixed operating points (MC vs formulas) ===")
    print(f"{'spec':<8} {'n_eff':>6} {'rho':>7} | {'q':>7} {'rho_eff':>9} "
          f"| {'MC':>9} {'(a)':>9} {'(b)':>9} {'(c)':>9} | best")
    grid_rows = []
    for kind in ("iso", "decay3", "rankdef"):
        lam = rmt_spectrum(kind)
        for n_eff in (2000, 1000, 500, 200, 100):
            w = 1.0 / n_eff
            for rho in (1e-3, 1e-2, 1e-1):
                q, rho_eff, v_a, v_b, v_c = rmt_formulas(lam, w, rho)
                mc = rmt_mc_fixed(kind, w, rho)
                errs = {"a": abs(mc - v_a), "b": abs(mc - v_b),
                        "c": abs(mc - v_c)}
                best = min(errs, key=errs.get)
                print(f"{kind:<8} {n_eff:>6} {rho:>7} | {q:>7.4f} "
                      f"{rho_eff:>9.4g} | {mc:>9.5f} {v_a:>9.5f} "
                      f"{v_b:>9.5f} {v_c:>9.5f} | {best}")
                grid_rows.append({"spectrum": kind, "full_rank": kind != "rankdef",
                                  "n_eff": n_eff, "omega": w, "rho": rho,
                                  "q": q, "rho_eff": rho_eff, "mc": mc,
                                  "formula_a": v_a, "formula_b": v_b,
                                  "formula_c": v_c, "best": best,
                                  "rel_err_b": abs(mc - v_b) / abs(mc)})
    pd.DataFrame(grid_rows).to_csv(RES / "rmt_grid.csv", index=False)
    print("saved results/rmt_grid.csv")
    print()
    print("=== operating points: the arm's converged control pair ===")
    from rls_baselines import make_prior as _mp
    rows = []
    print(f"{'spec':<8} {'w_conv':>8} {'rho_conv':>9} "
          f"{'MSE arm':>9} {'MSE DE':>9} {'rel err':>8}")
    for kind in ("iso", "decay3", "rankdef"):
        lam = rmt_spectrum(kind)
        ws, rhos, mses = [], [], []
        for seed in range(5):
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=RMT_D, T=RMT_T, sigma=0.1, spectrum=kind,
                drift="none", noise="gauss")
            G_init = _mp("warm", x, y, w_true, t1, RMT_T_WARM, Q0, lam0)
            preds, rhos_t, ws_t = run_spar_kf(
                x, y, RMT_T_WARM, G_init=G_init, prediction_units=True,
                debias_variance_floor=True, refresh_every=10, resid_fast=30,
                return_diag=True)
            y_te = y[RMT_T_WARM:]
            tail = slice(-1000, None)
            mses.append(float(np.mean((preds[tail] - y_te[tail]) ** 2)))
            ws.append(float(np.mean(ws_t[tail])))
            rhos.append(float(np.mean(rhos_t[tail])))
        w_conv = float(np.mean(ws))
        rho_conv = float(np.mean(rhos))
        mse_arm = float(np.mean(mses))
        q, rho_eff, mse_de, Jt = rmt_de_mse(lam, w_conv, rho_conv)
        rel = abs(mse_de - mse_arm) / mse_arm
        print(f"{kind:<8} {w_conv:>8.5f} {rho_conv:>9.4f} "
              f"{mse_arm:>9.5f} {mse_de:>9.5f} {100*rel:>7.1f}%")
        rows.append({"spectrum": kind, "w": w_conv, "rho": rho_conv,
                     "mse_arm": mse_arm, "q": q, "rho_eff": rho_eff,
                     "mse_de": mse_de, "rel_err": rel, "Jt": Jt})
    pd.DataFrame(rows).to_csv(RES / "rmt_verify.csv", index=False)
    print("saved results/rmt_verify.csv")
    print()
    print("=== sensitivity validation (rho d s/d rho): numerical vs closed ===")
    for kind in ("iso", "decay3", "rankdef"):
        lam = rmt_spectrum(kind)
        for w, rho in ((1/2000, 0.001), (1/500, 0.01), (1/200, 0.05)):
            num, closed = rmt_sens_check(lam, w, rho)
            print(f"{kind:<8} w={w:.5f} rho={rho}: "
                  f"num {num:.6f}  closed {closed:.6f}  "
                  f"rel {abs(num-closed)/abs(num)*100:.3f}%")
    print()
    diag_rmt_conflict()


# ================================================== alternating-trust regime
ALT_PHASE = 500       # switch period between revisit and drift phases
ALT_RHO_PINS = (1e-3, 0.01, 0.1, 1.0)


def make_alt_trust_stream(seed=0, D=128, T=10000, sigma=0.1,
                          spectrum="iso"):
    """Alternating-trust stream: the target switches between a noisy
    revisit of the prior (w0, where a large anchor pull is optimal) and a
    drifted target (w1, where a small pull is optimal) every ALT_PHASE
    steps after the warm-up. Features as in the grid generator."""
    rng = np.random.default_rng(seed)
    if spectrum == "iso":
        lam0 = np.ones(D)
    elif spectrum == "decay3":
        lam0 = np.logspace(0, -3, D)
    else:
        raise ValueError(spectrum)
    Q0, _ = np.linalg.qr(rng.normal(size=(D, D)))
    Z = rng.normal(size=(T, D))
    X = (Z * np.sqrt(lam0)[None, :]) @ Q0.T
    w0 = rng.normal(scale=0.5, size=D)
    w1 = w0 + rng.normal(scale=0.7, size=D)
    w_true = np.zeros((T, D))
    T_warm = 2000
    phase = np.zeros(T, dtype=int)
    for t in range(T_warm, T):
        phase[t] = ((t - T_warm) // ALT_PHASE) % 2
    w_true[:] = np.where(phase[:, None] == 0, w0, w1)
    # small per-phase jitter so the revisit is noisy, not exact
    jitter = rng.normal(scale=0.02, size=(T, D))
    eps = sigma * rng.normal(size=T)
    y = np.einsum("td,td->t", X, w_true + jitter) + eps
    return X, y, w0, Q0, lam0


def diag_alt_trust() -> None:
    """Adaptive anchor strength vs every fixed strength on the
    alternating-trust regime: the synthetic setting where the strength
    estimator matters on average."""
    T_warm, D, T = 2000, 128, 10000
    rows = []
    t0 = time.perf_counter()
    for spectrum in ("iso", "decay3"):
        for seed in range(5):
            x, y, w0, Q0, lam0 = make_alt_trust_stream(
                seed=seed, D=D, T=T, sigma=0.1, spectrum=spectrum)
            # warm prior: ridge fit on the warm-up (target = w0 there)
            A_w = x[:T_warm].T @ x[:T_warm] + 0.01 * T_warm * np.eye(D)
            G_init = np.linalg.solve(A_w, x[:T_warm].T @ y[:T_warm])
            y_te = y[T_warm:]
            arms = {"spar_kf(adaptive)": None}
            for rho in ALT_RHO_PINS:
                arms[f"spar_kf[rho_pin={rho:g}]"] = rho
            for name, rho_pin in arms.items():
                kw = ({} if rho_pin is None
                      else {"rho_pin": rho_pin})
                preds = run_spar_kf(
                    x, y, T_warm, G_init=G_init, refresh_every=10,
                    resid_fast=30, rho_drift_floor=0.02,
                    debias_variance_floor=True, **kw)[0]
                rows.append({
                    "spectrum": spectrum, "seed": seed, "method": name,
                    "mse_overall": float(np.mean((preds - y_te) ** 2)),
                })
            print(f"[alt_trust] {spectrum} s{seed} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "alt_trust.csv", index=False)
    print()
    print(df.groupby("method").mse_overall.agg(["mean", "max"])
          .round(4).sort_values("mean").to_string())
    print("saved results/alt_trust.csv")


# ============================================== finite-cadence suboptimality
def diag_subopt() -> None:
    """Relative gap between the surrogate cubic window target and the exact
    cycle-averaged optimum over the runtime cadence range."""
    k, sigma2, D = 10, 0.01, 128
    rows = []
    for k_omega in (0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        w0 = k_omega / k
        for s in (0.02, 0.05, 0.1, 0.2, 0.3):
            c = sigma2 * D * (1 - s) ** 2
            nu = float(np.sqrt((c * w0 ** 3 / 4) / D))
            w = np.linspace(w0 * 0.05, min(0.9, w0 * 20), 200000)
            r = (1 - w) ** k
            a = s * (2 - s)
            psi = 1 - 2 * s * (1 + s - s * s) * (1 - r) / ((1 + a) * (1 - a * r))
            dk = s * nu * k * (a / (1 - a) + 0.5)
            risk = sigma2 * D * (w / (2 - w)) * psi + D * (nu / w + dk) ** 2
            gap = risk[np.argmin(np.abs(w - w0))] / risk.min() - 1
            rows.append({"k_omega": k_omega, "omega": w0, "s": s,
                         "rel_gap": float(gap)})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "window_subopt.csv", index=False)
    print(df.pivot_table(index="k_omega", columns="s",
                         values="rel_gap").round(5).to_string())
    print("\nmax relative gap by s:")
    print(df.groupby("s").rel_gap.max().round(5).to_string())
    print("saved results/window_subopt.csv")


# ================================================================= dispatch
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", default="all",
                        choices=["vff", "windup", "sgd", "coupling",
                                 "rmt", "alt_trust", "subopt", "all"])
    args = parser.parse_args()
    which = (["vff", "windup", "sgd", "coupling", "rmt", "alt_trust",
              "subopt"]
             if args.diag == "all" else [args.diag])
    fns = {"vff": diag_vff, "windup": diag_windup, "sgd": diag_sgd,
           "coupling": diag_coupling, "rmt": diag_rmt,
           "alt_trust": diag_alt_trust, "subopt": diag_subopt}
    for name in which:
        print(f"\n=== diagnostics: {name} ===", flush=True)
        fns[name]()


if __name__ == "__main__":
    main()
