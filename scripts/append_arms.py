"""Consolidated appended-grid-arms driver.

Replaces the per-family append scripts with a single resumable driver: one
registry of arms, one loop over configurations and seeds, one row schema.
Every arm appends to the same factorial CSVs the original scripts used
(the per-drift grid files for the stream arms, the extras file for the
rest), so existing rows are skipped and no output changes.

Arms (destination: split = the per-drift grid files on the main suite,
extras = results/factorial_extras.csv):
  spar_kf2, kf_est, leaky_batch, spar_eg        split   (main + extra suites)
  leaky_batch[0.97], leaky_batch[0.95]          extras  (extended sweep)
  rls_sym[...]                                  extras  (symmetrized covariance RLS)
  kf_est_slow                                   extras  (main)
  vff[...]  (six members)                       extras  (main)
  ogd_profile, vff_info[...]                    extras  (main)
  ogd_profile_d                                extras  (main)
  spar_eg_plus, kf_causal[q*=...]               extras  (main + extra)
  kf_est_causal, kf_est_slow_causal             extras  (main)
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

from ogd_spar_test import run_ogd_profile, run_ogd_spar  # noqa: E402
from rls_baselines import (make_prior, make_stream_fac, run_kf,  # noqa: E402
                           run_kf_est, run_leaky_batch, run_spar_eg,
                           run_spar_kf)
from synthetic_factorial import phases_for  # noqa: E402

RES = ROOT / "results"

MAIN = [(d, s, p, "gauss")
        for d in ("abrupt", "covrot", "both")
        for s in ("iso", "decay3", "rankdef")
        for p in ("oracle", "warm", "adversarial")]
EXTRA = [("exdeath", "decay3", "warm", "gauss"), ("osc", "iso", "warm", "gauss"),
         ("covrot", "decay3", "warm", "t3"), ("none", "iso", "warm", "gauss")]

SPLIT_METHODS = ("spar_kf2", "kf_est[q=", "leaky_batch[", "spar_eg")


def run_vff_info(x, y, T_warm, lam0=1.0, lam_min=0.95, ridge_frac=0.01):
    """Variable forgetting factor RLS in information form: the per-step
    forgetting factor follows Fortescue et al. (1981), but the weighted
    least-squares objective is solved in batch (information form) with a
    matched ridge, as in leaky_batch. No covariance recursion anywhere."""
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    r_warm = y[:T_warm] - x[:T_warm] @ np.linalg.solve(
        A + n_eff * ridge_frac * np.eye(D), B)
    sigma0 = float(np.var(r_warm)) + 1e-12
    alpha = 1.0
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        phi = x[t]
        pred = phi @ np.linalg.solve(
            A + n_eff * ridge_frac * np.eye(D), B)
        preds[t - T_warm] = pred
        e = y[t] - pred
        lam = float(np.clip(1.0 - (e * e) / (lam0 * sigma0), lam_min, 1.0))
        alpha = 1.0 - (1.0 - lam)
        A = alpha * A + np.outer(phi, phi)
        B = alpha * B + phi * y[t]
        n_eff = alpha * n_eff + 1.0
    return preds


def run_rls_sym(x, y, T_warm, lam, rho=0.1):
    """Covariance-form leaky RLS with per-step symmetrization of the
    iterated inverse (the numerically protected recursion of the windup
    mechanism note)."""
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
        P = 0.5 * (P + P.T)
    return preds


def run_kf_causal(x, y, T_warm, t1, qs=(1e-6, 1e-5, 1e-4)):
    """Random-walk Kalman with q selected causally on the stationary
    pre-drift segment only; returns (method name, predictions)."""
    y_te = y[T_warm:]
    pre = slice(0, max(t1 - T_warm, 1))
    cand = []
    for q in qs:
        preds_q = run_kf(x, y, T_warm, q=q, R=0.01)
        cand.append((float(np.mean((preds_q[pre] - y_te[pre]) ** 2)),
                     q, preds_q))
    _, q_star, preds = min(cand)
    return f"kf_causal[q*={q_star:g}]", preds


# ---------------------------------------------------------------- registry
# Each arm: (method, dest, universe, needs_prior, runner)
# runner: (x, y, T_warm, G_init, ctx) -> (name, preds) or list of such.

def _arm(method, dest, universe, prior, runner):
    return {"method": method, "dest": dest, "universe": universe,
            "prior": prior, "runner": runner}


def _simple(method, dest, universe, prior, fn):
    def runner(x, y, T_warm, G_init, ctx):
        preds = fn(x, y, T_warm, G_init)
        return [(method, preds)]
    return _arm(method, dest, universe, prior, runner)


ARMS = [
    # stream arms (per-drift grid files on the main suite, extras otherwise)
    # headline prediction-unit arm (the published SPAR-Pull); `spar_kf2` is
    # kept only as the legacy parameter-unit ablation.
    _simple("spar_pull", "split", "main+extra", True,
            lambda x, y, w, g: run_spar_kf(
                x, y, w, G_init=g, refresh_every=10, resid_fast=30,
                rho_drift_floor=0.02, debias_variance_floor=True,
                prediction_units=True)[0]),
    _simple("spar_kf2", "split", "main+extra", True,
            lambda x, y, w, g: run_spar_kf(
                x, y, w, G_init=g, refresh_every=10, resid_fast=30,
                rho_drift_floor=0.02, debias_variance_floor=True)[0]),
    _simple("kf_est[q=1e-4]", "split", "main+extra", False,
            lambda x, y, w, g: run_kf_est(x, y, w, q=1e-4)),
    _simple("kf_est[q=1e-5]", "split", "main+extra", False,
            lambda x, y, w, g: run_kf_est(x, y, w, q=1e-5)),
    _simple("leaky_batch[0.997]", "split", "main+extra", False,
            lambda x, y, w, g: run_leaky_batch(x, y, w, lam=0.997)),
    _simple("leaky_batch[0.995]", "split", "main+extra", False,
            lambda x, y, w, g: run_leaky_batch(x, y, w, lam=0.995)),
    _simple("leaky_batch[0.99]", "split", "main+extra", False,
            lambda x, y, w, g: run_leaky_batch(x, y, w, lam=0.99)),
    _simple("spar_eg", "split", "main+extra", True,
            lambda x, y, w, g: run_spar_eg(x, y, w, G_init=g)),
    # extended stable-window sweep (information form)
    _simple("leaky_batch[0.97]", "extras", "main", False,
            lambda x, y, w, g: run_leaky_batch(x, y, w, lam=0.97)),
    _simple("leaky_batch[0.95]", "extras", "main", False,
            lambda x, y, w, g: run_leaky_batch(x, y, w, lam=0.95)),
    # symmetrized covariance-form RLS (the protected recursion)
    _simple("rls_sym[0.99]", "extras", "main", False,
            lambda x, y, w, g: run_rls_sym(x, y, w, lam=0.99)),
    _simple("rls_sym[0.97]", "extras", "main", False,
            lambda x, y, w, g: run_rls_sym(x, y, w, lam=0.97)),
    _simple("rls_sym[0.95]", "extras", "main", False,
            lambda x, y, w, g: run_rls_sym(x, y, w, lam=0.95)),
    # slow-horizon estimated-R Kalman
    _simple("kf_est_slow[q=1e-4]", "extras", "main", False,
            lambda x, y, w, g: run_kf_est(x, y, w, q=1e-4, eta_R=5e-5)),
    _simple("kf_est_slow[q=1e-5]", "extras", "main", False,
            lambda x, y, w, g: run_kf_est(x, y, w, q=1e-5, eta_R=5e-5)),
    # covariance-form VFF, both calibrations x both floors
    _simple("vff[fortescue]", "extras", "main", False,
            lambda x, y, w, g: _run_vff_cov(x, y, w, lam0=1.0,
                                            lam_min=0.95)),
    _simple("vff[fortescue,lmin=0.99]", "extras", "main", False,
            lambda x, y, w, g: _run_vff_cov(x, y, w, lam0=1.0,
                                            lam_min=0.99)),
    _simple("vff[fortescue,N0=100]", "extras", "main", False,
            lambda x, y, w, g: _run_vff_cov(x, y, w, lam0=100.0)),
    _simple("vff[fortescue,N0=100,lmin=0.99]", "extras", "main", False,
            lambda x, y, w, g: _run_vff_cov(x, y, w, lam0=100.0,
                                            lam_min=0.99)),
    _simple("vff[fortescue,N0=1000]", "extras", "main", False,
            lambda x, y, w, g: _run_vff_cov(x, y, w, lam0=1000.0)),
    _simple("vff[fortescue,N0=1000,lmin=0.99]", "extras", "main", False,
            lambda x, y, w, g: _run_vff_cov(x, y, w, lam0=1000.0,
                                           lam_min=0.99)),
    # gradient arms
    _simple("ogd_profile", "extras", "main", True,
            lambda x, y, w, g: run_ogd_spar(x, y, w, G_init=g)),
    _simple("vff_info[fortescue]", "extras", "main", False,
            lambda x, y, w, g: run_vff_info(x, y, w)),
    _simple("vff_info[fortescue,lmin=0.99]", "extras", "main", False,
            lambda x, y, w, g: run_vff_info(x, y, w, lam_min=0.99)),
    _simple("ogd_profile_d", "extras", "main", True,
            lambda x, y, w, g: run_ogd_profile(x, y, w, G_init=g)),
    # SPAR-EG+ (needs the SPAR-Pull trajectory) and the causal-q Kalman
    _arm("spar_eg_plus", "extras", "main+extra", True,
         lambda x, y, T_warm, G_init, ctx: [
             ("spar_eg_plus", run_spar_eg(
                 x, y, T_warm, G_init=G_init,
                 extra_preds=ctx["kf_preds"](x, y, T_warm, G_init)))]),
    _arm("kf_causal", "extras", "main+extra", False,
         lambda x, y, T_warm, G_init, ctx: [
             run_kf_causal(x, y, T_warm, ctx["t1"])]),
    # causally-tuned estimated-R Kalman arms
    _simple("kf_est_causal[q=1e-06]", "extras", "main", False,
            lambda x, y, w, g: run_kf_est(x, y, w, q=1e-6, eta_R=0.01)),
    _simple("kf_est_causal[q=1e-05]", "extras", "main", False,
            lambda x, y, w, g: run_kf_est(x, y, w, q=1e-5, eta_R=0.01)),
    _simple("kf_est_slow_causal[q=1e-06]", "extras", "main", False,
            lambda x, y, w, g: run_kf_est(x, y, w, q=1e-6, eta_R=5e-5)),
    _simple("kf_est_slow_causal[q=1e-05]", "extras", "main", False,
            lambda x, y, w, g: run_kf_est(x, y, w, q=1e-5, eta_R=5e-5)),
]


def _run_vff_cov(x, y, T_warm, lam0=1.0, lam_min=0.95, rho=0.1):
    """Covariance-form VFF-RLS (Fortescue et al., 1981), the standard
    recursion, at the given calibration (lam0 = N0 or 1) and floor."""
    from rls_baselines import run_vff
    return run_vff(x, y, T_warm, lam0=lam0, lam_min=lam_min, rho=rho)


def _kf_preds_cache():
    """Lazily run the SPAR-Pull trajectory once per (config, seed) and
    reuse it for SPAR-EG+ (the sixth-expert pool)."""
    store = {}

    def get(x, y, T_warm, G_init):
        key = id(x)
        if key not in store:
            store[key] = run_spar_kf(
                x, y, T_warm, G_init=G_init, refresh_every=10,
                resid_fast=30, rho_drift_floor=0.02,
                debias_variance_floor=True, prediction_units=True)[0]
        return store[key]
    return get


def load_done(dest_csv):
    """Per-method done sets from one CSV: (spec, drift, prior, noise,
    seed, method). The kf_causal arm writes a dynamic method name per
    run (kf_causal[q*=...]); both the full name and the registry name
    are recorded so the arm-level check matches."""
    done = set()
    if not dest_csv.exists():
        return done
    prev = pd.read_csv(dest_csv)
    for _, r in prev.iterrows():
        m = str(r["method"])
        key = (r["spectrum"], r["drift"], r["prior"], r["noise"],
               int(r["seed"]), m)
        done.add(key)
        if m.startswith("kf_causal"):
            done.add(key[:5] + ("kf_causal",))
    return done


def dest_path(arm, drift, noise):
    """Stream arms write to the per-drift grid files on the main suite
    (the three main drifts, Gaussian noise); everything else (the extra
    suite, including the t3 covrot stress config) writes to extras."""
    if (arm["dest"] == "split" and drift in ("abrupt", "covrot", "both")
            and noise == "gauss"):
        return RES / f"factorial_d_{drift}.csv"
    return RES / "factorial_extras.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--T", type=int, default=10000)
    parser.add_argument("--T_warm", type=int, default=2000)
    args = parser.parse_args()

    done = {}
    for csv in {RES / "factorial_extras.csv",
                RES / "factorial_d_abrupt.csv",
                RES / "factorial_d_covrot.csv",
                RES / "factorial_d_both.csv"}:
        done[str(csv)] = load_done(csv)

    universes = {"main": MAIN, "extra": EXTRA,
                 "main+extra": MAIN + EXTRA}
    pending = []
    for arm in ARMS:
        for combo in universes[arm["universe"]]:
            for seed in range(args.seeds):
                drift, spectrum, prior, noise = combo
                key = (spectrum, drift, prior, noise, seed, arm["method"])
                if key in done[str(dest_path(arm, drift, noise))]:
                    continue
                pending.append((combo, seed, arm))

    print(f"[append_arms] pending work items: {len(pending)}", flush=True)
    if not pending:
        print("[append_arms] nothing to do (all rows present)")
        return

    t0 = time.perf_counter()
    written = 0
    by_combo = {}
    for combo, seed, arm in pending:
        by_combo.setdefault((combo, seed), []).append(arm)

    for (combo, seed), arms in by_combo.items():
        drift, spectrum, prior, noise = combo
        x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
            seed=seed, D=args.D, T=args.T, sigma=0.1, spectrum=spectrum,
            drift=drift, noise=noise)
        G_init = (make_prior(prior, x, y, w_true, t1, args.T_warm, Q0,
                             lam0)
                  if any(a["prior"] for a in arms) else None)
        y_te = y[args.T_warm:]
        n_te = len(y_te)
        regs = phases_for(drift, t1 - args.T_warm, t2 - args.T_warm, n_te)
        ctx = {"t1": t1, "kf_preds": _kf_preds_cache()}
        rows = []
        for arm in arms:
            for name, preds in arm["runner"](x, y, args.T_warm, G_init,
                                             ctx):
                row = {"spectrum": spectrum, "drift": drift,
                       "prior": prior, "noise": noise, "seed": seed,
                       "method": name, "D": args.D, "T": args.T,
                       "wall_s": time.perf_counter() - t0}
                for reg_name, reg in regs.items():
                    row[f"mse_{reg_name}"] = float(
                        np.mean((preds[reg] - y_te[reg]) ** 2)) \
                        if reg.stop > reg.start else float("nan")
                row["mse_overall"] = float(
                    np.mean((preds - y_te) ** 2))
                row["recovery"] = float("nan")
                row["diverged"] = bool(row["mse_overall"] > 1e3
                                       or not np.all(np.isfinite(preds)))
                row["_dest"] = str(dest_path(arm, drift, noise))
                rows.append(row)
        for dest in {r["_dest"] for r in rows}:
            sub = [r for r in rows if r["_dest"] == dest]
            for r in sub:
                r.pop("_dest")
            pd.DataFrame(sub).to_csv(Path(dest), mode="a",
                                     header=not Path(dest).exists(),
                                     index=False)
            written += len(sub)
        print(f"[append_arms] {drift}/{spectrum}/{prior} s{seed} "
              f"({time.perf_counter() - t0:.0f}s, {written} rows)",
              flush=True)
    print(f"[append_arms] done ({written} new rows, "
          f"{time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
