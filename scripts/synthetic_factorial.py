"""Tier-1 synthetic factorial: multi-factor grid with all baselines.

Axes:
  spectrum x {iso, decay3, rankdef}
  drift    x {abrupt, covrot, both}        (main grid)
             + {none, exdeath, osc} + heavy-tail noise (extra suite)
  prior    x {oracle, warm, adversarial}
  seeds    x 5

Methods (all closed-form or first-order online, all predict-then-update):
  warm, no_anchor, leaky RLS (lam sweep), directional forgetting DF
  (lam_f sweep), AROW (r sweep), Kalman (q sweep, true-noise R provided),
  SGD/Adam (lr sweep), WiSE-FT (alpha envelope, zero-shot = the prior),
  SPAR scalar-rho (v2.1), SPAR diagonal-rho (theory-exact, NEW).

Sweeps are reported per-value; the aggregator takes the per-config best
(the tuned-baseline treatment, which favours the baselines).
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rls_baselines import (make_prior, make_stream_fac, run_adam,  # noqa: E402
                           run_arow, run_df, run_kf, run_rls,
                           run_spar_diag, run_sgd, wiseft_models,
                           wiseft_preds)
from spar_test import run_spar_v2  # noqa: E402

LEAKY_LAMS = (0.999, 0.997, 0.995, 0.99)
DF_LAMS = (0.98, 0.95, 0.90)
AROW_RS = (0.05, 0.3)
KF_QS = (1e-6, 1e-5, 1e-4)
SGD_LRS = (0.003, 0.01)
WISEFT_ALPHAS = np.round(np.linspace(0.0, 1.0, 11), 2)


def phases_for(drift, t1, t2, n_te):
    """Region slices into pre / during / post (prediction-time indexing)."""
    if drift in ("abrupt", "both"):
        rec = 500
        return {"pre": slice(0, t1),
                "dur": slice(t1, t1 + rec),
                "post": slice(t1 + rec, n_te)}
    if drift in ("covrot", "exdeath", "gradual"):
        return {"pre": slice(0, t1),
                "dur": slice(t1, t2),
                "post": slice(t2, n_te)}
    if drift == "osc":
        return {"pre": slice(0, t1),
                "dur": slice(t1, n_te),
                "post": slice(t1, n_te)}
    # none
    return {"pre": slice(0, n_te), "dur": slice(0, 0), "post": slice(0, 0)}


def recovery_time(preds, y_te, t1, pre_mse, window=100, cap=None):
    if pre_mse is None or not np.isfinite(pre_mse) or pre_mse <= 0:
        return float("nan")
    n = len(preds)
    cap = cap if cap is not None else n - t1
    for tau in range(0, min(n - t1 - window, cap)):
        end = t1 + tau + window
        if end > n:
            break
        if np.mean((preds[t1 + tau:end] - y_te[t1 + tau:end]) ** 2) <= 2.0 * pre_mse:
            return float(tau)
    return float("nan") if (n - t1) > cap else float(n - t1)


def run_all_methods(x, y, t1, t2, Q0, lam0, w_true, T_warm, sigma2_true,
                    G_init, drift):
    """Returns dict method -> preds."""
    y_te = y[T_warm:]
    n_te = len(y_te)
    out = {}

    # warm (frozen)
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W_warm = np.linalg.solve(A_w + 0.01 * T_warm * np.eye(x.shape[1]), b_w)
    out["warm"] = x[T_warm:] @ W_warm

    # no_anchor (batch count-scaled ridge, no forgetting)
    A = A_w.astype(np.float64).copy()
    B = b_w.astype(np.float64).copy()
    n_eff = float(T_warm)
    pa = np.zeros(n_te)
    for t in range(T_warm, len(y)):
        W = np.linalg.solve(A + n_eff * 0.01 * np.eye(x.shape[1]), B)
        pa[t - T_warm] = x[t] @ W
        A += np.outer(x[t], x[t])
        B += x[t] * y[t]
        n_eff += 1.0
    out["no_anchor"] = pa

    # swept baselines
    for lam in LEAKY_LAMS:
        out[f"leaky[{lam}]"] = run_rls(x, y, T_warm, lam=lam, rho=0.1)
    for lf in DF_LAMS:
        out[f"df[{lf}]"] = run_df(x, y, T_warm, lam_f=lf, rho=0.1)
    for r in AROW_RS:
        out[f"arow[{r}]"] = run_arow(x, y, T_warm, r=r)
    for q in KF_QS:
        out[f"kf[q={q}]"] = run_kf(x, y, T_warm, q=q, R=sigma2_true)
    for lr in SGD_LRS:
        out[f"sgd[{lr}]"] = run_sgd(x, y, T_warm, lr=lr)
        out[f"adam[{lr}]"] = run_adam(x, y, T_warm, lr=lr)

    # WiSE-FT (alpha envelope; zero-shot = the prior)
    D = x.shape[1]
    if G_init is not None:
        W_zero = np.asarray(G_init, dtype=np.float64)
        _, W_ft = wiseft_models(x, y, T_warm)
    else:
        W_zero, W_ft = wiseft_models(x, y, T_warm)
    best_a, best_mse = 0.0, np.inf
    wiseft_pred_cache = {}
    for a in WISEFT_ALPHAS:
        p = wiseft_preds(x[T_warm:], W_zero, W_ft, float(a))
        wiseft_pred_cache[float(a)] = p
        m = float(np.mean((p - y_te) ** 2))
        if m < best_mse:
            best_mse, best_a = m, float(a)
    out[f"wiseft[a={best_a}]"] = wiseft_pred_cache[best_a]
    out["wiseft[grid]"] = None  # handled at aggregation via cache
    wiseft_grid = wiseft_pred_cache

    # SPAR arms (scalar v2.1 + diagonal)
    out["spar_scalar"] = run_spar_v2(x, y, T_warm, rho_init=0.1, alpha=0.997,
                                     adaptive_rho=True, refresh=True,
                                     G_init=G_init)[0]
    out["spar_diag"] = run_spar_diag(x, y, T_warm, G_init=G_init, alpha=0.997)
    out["spar_scalar_a1"] = run_spar_v2(x, y, T_warm, rho_init=0.1, alpha=1.0,
                                         adaptive_rho=True, refresh=True,
                                         G_init=G_init)[0]
    return out, wiseft_grid


def append_rows(rows, out_csv):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)


def done_keys(out_csv):
    import pandas as pd
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    return set(zip(df["spectrum"], df["drift"], df["prior"], df["noise"],
                   df["seed"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="main", choices=["main", "extra"])
    parser.add_argument("--drifts", default=None,
                        help="comma list to override (e.g. abrupt,covrot)")
    parser.add_argument("--spectra", default=None)
    parser.add_argument("--priors", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--T", type=int, default=10000)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.suite == "main":
        drifts = args.drifts.split(",") if args.drifts \
            else ["abrupt", "covrot", "both"]
        spectra = args.spectra.split(",") if args.spectra \
            else ["iso", "decay3", "rankdef"]
        priors = args.priors.split(",") if args.priors \
            else ["oracle", "warm", "adversarial"]
        noises = ["gauss"]
        combos = [(d, s, p, "gauss")
                  for d in drifts for s in spectra for p in priors]
    else:
        drifts = args.drifts.split(",") if args.drifts \
            else ["exdeath", "osc", "covrot", "none"]
        spectra = args.spectra.split(",") if args.spectra \
            else ["decay3", "iso", "decay3", "iso"]
        priors = args.priors.split(",") if args.priors \
            else ["warm", "warm", "warm", "warm"]
        noises = ["gauss", "gauss", "t3", "gauss"]
        combos = list(zip(drifts, spectra, priors, noises))

    T_warm = 2000
    rows = []
    t_start = time.perf_counter()
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = args.tag or args.suite
    out_csv = out_dir / f"factorial_{tag}.csv"
    done = done_keys(out_csv)
    if done:
        print(f"resuming: {len(done)} (config,seed) combos already in {out_csv}",
              flush=True)
    for drift, spectrum, prior, noise in combos:
        for seed in range(args.seeds):
            if (spectrum, drift, prior, noise, seed) in done:
                continue
            t0 = time.perf_counter()
            x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
                seed=seed, D=args.D, T=args.T, sigma=args.sigma,
                spectrum=spectrum, drift=drift, noise=noise)
            if t1 <= T_warm:
                raise RuntimeError("t1 must exceed T_warm")
            t1p, t2p = t1 - T_warm, t2 - T_warm
            G_init = make_prior(prior, x, y, w_true, t1, T_warm, Q0, lam0)
            y_te = y[T_warm:]
            n_te = len(y_te)
            regs = phases_for(drift, t1p, t2p, n_te)

            preds_all, wiseft_grid = run_all_methods(
                x, y, t1, t2, Q0, lam0, w_true, T_warm,
                sigma2_true=args.sigma ** 2, G_init=G_init, drift=drift)

            # pre-mse reference for recovery
            ref = preds_all["no_anchor"]
            pre_ref = float(np.mean((ref[regs["pre"]] - y_te[regs["pre"]]) ** 2))

            for name, preds in preds_all.items():
                if preds is None:
                    continue
                row = {
                    "spectrum": spectrum, "drift": drift, "prior": prior,
                    "noise": noise, "seed": seed, "method": name,
                    "D": args.D, "T": args.T,
                    "wall_s": time.perf_counter() - t0,
                }
                for reg_name, reg in regs.items():
                    if reg.stop > reg.start:
                        row[f"mse_{reg_name}"] = float(
                            np.mean((preds[reg] - y_te[reg]) ** 2))
                    else:
                        row[f"mse_{reg_name}"] = float("nan")
                row["mse_overall"] = float(np.mean((preds - y_te) ** 2))
                if drift in ("abrupt", "both"):
                    row["recovery"] = recovery_time(
                        preds, y_te, t1p, row.get("mse_pre", pre_ref))
                else:
                    row["recovery"] = float("nan")
                row["diverged"] = bool(row["mse_overall"] > 1e3
                                       or not np.all(np.isfinite(preds)))
                rows.append(row)

            # WiSE-FT full grid rows (aggregator picks per-phase best)
            for a, p in wiseft_grid.items():
                row = {
                    "spectrum": spectrum, "drift": drift, "prior": prior,
                    "noise": noise, "seed": seed, "method": f"wiseftgrid[{a}]",
                    "D": args.D, "T": args.T, "wall_s": 0.0,
                }
                for reg_name, reg in regs.items():
                    row[f"mse_{reg_name}"] = float(
                        np.mean((p[reg] - y_te[reg]) ** 2)) \
                        if reg.stop > reg.start else float("nan")
                row["mse_overall"] = float(np.mean((p - y_te) ** 2))
                row["recovery"] = float("nan")
                row["diverged"] = bool(row["mse_overall"] > 1e3)
                rows.append(row)

            append_rows(rows, out_csv)
            rows = []
            print(f"[{drift}/{spectrum}/{prior}/{noise}] seed {seed} done "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"all done -> {out_csv} "
          f"({time.perf_counter() - t_start:.0f}s total this session)")


if __name__ == "__main__":
    main()
