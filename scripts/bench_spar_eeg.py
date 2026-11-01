"""SPAR on real EEG: BCI IV 2a cross-session streaming classification.

Tests the imperfect-prior regime of the SPAR theory on real data: the
anchor is the warm-start from the calibration session (NOT an oracle),
the stream is short (288 trials), and drift is physiological
(non-stationarity within/between sessions).

Arms (all closed-form, predict-then-update, supervised online):
  warm         : no streaming (upper reference for frozen models)
  pam_g0.1     : existing fixed-anchor arm  (gamma=0.1, ridge=0.01, alpha=1)
  pam_g1.0     : same with gamma=1 (the gamma=1 design principle)
  leaky_aX     : pure forgetting ridge, no anchor (gamma=0)
  spar_fix     : scalar-rho anchored solve, pull-identity refresh, gamma=1
  spar_adp     : + adaptive rho* (reference-conflict + residual CUSUM)
  spar_full    : + accumulator forgetting alpha<1
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys
import time

import jax
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_eeg import (FEAT_CFG, apply_whitener, bandpass, cov_features,  # noqa: E402
                       downsample, ea_whitener, feat_eval)
from core.metrics import balanced_accuracy, cohen_kappa  # noqa: E402
from dataio import load_eeg_subject  # noqa: E402
from dataio.eeg import CUE_NAMES  # noqa: E402
from runner import classifier  # noqa: E402


def onehot(y, K):
    Y = np.zeros((len(y), K), dtype=np.float64)
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return Y


def stream_spar(X_tr, y_tr, X_st, y_st, K, rho_init=0.1, gamma=1.0, alpha=1.0,
                adaptive=True, refresh_every=10, ref_horizon=150,
                ridge=0.01, K_cal=2, eta_C=0.2, resid_fast=30,
                return_scores=False):
    """Multi-class SPAR streaming head (v2.1 rules, K output columns)."""
    D = X_tr.shape[1]
    A = X_tr.T @ X_tr.astype(np.float64)
    B = X_tr.T @ onehot(y_tr, K)
    n = float(len(X_tr))

    G = np.linalg.solve(A + n * ridge * np.eye(D), B)
    G_ref = G.copy()
    resid_w = onehot(y_tr, K) - X_tr @ G
    sigma2 = float(np.mean(resid_w ** 2)) + 1e-12
    sigma2_fast = sigma2_slow = sigma2
    Abar_warm = A / n
    tr_inv_warm = float(np.trace(np.linalg.inv(Abar_warm + 1e-8 * np.eye(D))))
    C_floor = 0.5 * sigma2 * D * K / n
    C_ema = 2 * C_floor
    rho = rho_init

    Yst = onehot(y_st, K)
    eta_f = 1.0 / resid_fast
    eta_s = 1.0 / ref_horizon
    eta_ref = refresh_every / ref_horizon

    preds = np.zeros(len(y_st), dtype=int)
    scores = np.zeros((len(y_st), K))
    rhos = []
    step = 0
    cyc = 0

    for i in range(len(y_st)):
        x = X_st[i]
        if adaptive and step % refresh_every == 0:
            Abar = A / n
            mu_bar = float(np.trace(Abar) / D) + 1e-12
            mu_all = np.linalg.eigvalsh(Abar)
            d_act = max(1, int(np.sum(mu_all > 1e-8 * max(float(mu_all.max()), 1e-12))))
            V_hat = sigma2_slow * d_act * K / n
            W_data = np.linalg.solve(A + n * 1e-6 * np.eye(D), B)
            delta = G_ref - W_data
            C_ref = float(np.sum(delta * (Abar @ delta))) - V_hat
            E_resid = max(0.0, sigma2_fast - sigma2_slow)
            C_resid = E_resid
            C_floor_use = 0.5 * sigma2_slow * d_act * K / n
            C_raw = max(C_ref, C_resid, C_floor_use, 1e-10)
            C_ema = (1 - eta_C) * C_ema + eta_C * C_raw
            rho_raw = mu_bar * V_hat / C_ema
            cap = rho_init if cyc < K_cal else 2.0
            rho = float(np.clip(rho_raw, 1e-5, cap))
            rhos.append(rho)

        W = np.linalg.solve(A + n * rho * np.eye(D), B + n * rho * gamma * G)
        s_i = x @ W
        scores[i] = s_i
        preds[i] = int(np.argmax(s_i))

        e_t = float(np.mean((Yst[i] - s_i) ** 2))
        sigma2_fast = (1 - eta_f) * sigma2_fast + eta_f * e_t
        sigma2_slow = (1 - eta_s) * sigma2_slow + eta_s * e_t

        if step % refresh_every == 0 and step > 0:
            Abar = A / n
            mu, U = np.linalg.eigh(Abar)
            beta = (mu / (mu + rho))[:, None]
            Gp = U.T @ G
            Wp = U.T @ W
            G = U @ (beta * Wp + (1 - beta) * Gp)
            G_ref = (1 - eta_ref) * G_ref + eta_ref * W
            cyc += 1

        A = alpha * A + np.outer(x, x)
        B = alpha * B + np.outer(x, Yst[i])
        n = alpha * n + 1.0
        step += 1

    if return_scores:
        return preds, rhos, scores
    return preds, rhos


def leaky_scores(X_tr, y_tr, X_st, y_st, K, alpha=1.0, ridge=0.01):
    """Causal leaky-ridge expert returning per-trial class scores."""
    D = X_tr.shape[1]
    A = X_tr.T @ X_tr.astype(np.float64)
    B = X_tr.T @ onehot(y_tr, K)
    n = float(len(X_tr))
    Yst = onehot(y_st, K)
    scores = np.zeros((len(y_st), K))
    for i, x in enumerate(X_st):
        W = np.linalg.solve(A + n * ridge * np.eye(D), B)
        scores[i] = x @ W
        A = alpha * A + np.outer(x, x)
        B = alpha * B + np.outer(x, Yst[i])
        n = alpha * n + 1.0
    return scores


def hedge_mix(score_list, y_st, K):
    """Causal exponentially weighted mixture over expert score paths."""
    Kexp = len(score_list)
    pi = np.ones(Kexp) / Kexp
    preds = np.zeros(len(y_st), dtype=int)
    S = 1.0
    Yst = onehot(y_st, K)
    for i in range(len(y_st)):
        sc = np.array([s[i] for s in score_list])       # Kexp x K
        mix = pi @ sc
        preds[i] = int(np.argmax(mix))
        losses = np.mean((sc - Yst[i]) ** 2, axis=1)
        eta = min(0.5 / max(float(losses.max()), 1e-6),
                  np.sqrt(np.log(Kexp) / max(S, 1.0)))
        pi = pi * np.exp(-eta * losses)
        pi /= pi.sum()
        S += float(np.mean((mix - Yst[i]) ** 2))
    return preds


def stream_spar_eg(X_tr, y_tr, X_st, y_st, K, include_spar=True,
                   spar_scores=None):
    """SPAR-EG / SPAR-EG+: frozen anchor + leaky windows (+ SPAR-Pull)."""
    score_list = [
        leaky_scores(X_tr, y_tr, X_st, y_st, K, alpha=1.0),
        leaky_scores(X_tr, y_tr, X_st, y_st, K, alpha=0.995),
        leaky_scores(X_tr, y_tr, X_st, y_st, K, alpha=0.99),
        leaky_scores(X_tr, y_tr, X_st, y_st, K, alpha=0.98),
    ]
    D = X_tr.shape[1]
    A = X_tr.T @ X_tr.astype(np.float64)
    B = X_tr.T @ onehot(y_tr, K)
    n = float(len(X_tr))
    G = np.linalg.solve(A + n * 0.01 * np.eye(D), B)
    score_list.append(X_st @ G)
    if include_spar:
        if spar_scores is None:
            spar_scores = stream_spar(X_tr, y_tr, X_st, y_st, K,
                                      adaptive=True, alpha=1.0,
                                      return_scores=True)[2]
        score_list.append(spar_scores)
    return hedge_mix(score_list, y_st, K)


def main() -> None:
    t0 = time.perf_counter()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2a", choices=["2a", "2b"])
    parser.add_argument("--subjects", type=int, default=9)
    args = parser.parse_args()
    dataset = args.dataset
    n_subjects = args.subjects
    feature_fn = None
    from core import features
    feature_fn = features.make_feature_fn(FEAT_CFG)

    rows = []
    for subject in range(1, n_subjects + 1):
        out = load_eeg_subject(dataset=dataset, subject=subject)
        (X_tr, y_tr), (X_st, y_st) = out["train"], out["stream"]
        K = len(CUE_NAMES[dataset])

        # hybrid feature pipeline (the current best from bench_eeg)
        W_ea_tr = ea_whitener(X_tr)
        W_ea_st = ea_whitener(X_st)
        phi_tr_ea, phi_st_ea = feat_eval(
            apply_whitener(X_tr, W_ea_tr), apply_whitener(X_st, W_ea_st), feature_fn)
        Xbp_tr = bandpass(apply_whitener(X_tr, W_ea_tr))
        Xbp_st = bandpass(apply_whitener(X_st, W_ea_st))
        ref_cov = cov_features(Xbp_tr).mean(0)
        Fc_tr = cov_features(Xbp_tr, ref_cov)
        Fc_st = cov_features(Xbp_st, ref_cov)
        mc = Fc_tr.mean(0)
        sc = Fc_tr.std(0) + 1e-12
        H_tr = np.concatenate([phi_tr_ea, (Fc_tr - mc) / sc], axis=1)
        H_st = np.concatenate([phi_st_ea, (Fc_st - mc) / sc], axis=1)

        scores = {}

        # references
        W_w = classifier.fit_warm(H_tr, y_tr, K, ridge=0.01, gamma=0.0)
        pred = np.argmax(H_st @ W_w, axis=1)
        scores["warm"] = (balanced_accuracy(y_st, pred), cohen_kappa(y_st, pred))

        def add(name, preds):
            scores[name] = (balanced_accuracy(y_st, preds),
                            cohen_kappa(y_st, preds))

        # existing PAM arms
        add("pam_g0.1", classifier.stream(H_tr, y_tr, H_st, y_st, K, alpha=1.0,
                                          gamma=0.1, G=W_w, reanchor_every=10))
        add("pam_g1.0", classifier.stream(H_tr, y_tr, H_st, y_st, K, alpha=1.0,
                                          gamma=1.0, G=W_w, reanchor_every=10))

        # forgetting-only baselines
        for a in (0.995, 0.99, 0.98):
            add(f"leaky_a{a}", classifier.stream(H_tr, y_tr, H_st, y_st, K,
                                                  alpha=a, gamma=0.0, reanchor_every=10))

        # SPAR arms
        add("spar_fix", stream_spar(H_tr, y_tr, H_st, y_st, K, adaptive=False,
                                    rho_init=0.1, alpha=1.0)[0])
        pred_adp, _, scores_adp = stream_spar(
            H_tr, y_tr, H_st, y_st, K, adaptive=True, rho_init=0.1,
            alpha=1.0, return_scores=True)
        add("spar_adp", pred_adp)
        for a in (1.0, 0.995, 0.99):
            add(f"spar_full_a{a}", stream_spar(H_tr, y_tr, H_st, y_st, K,
                                               adaptive=True, rho_init=0.1,
                                               alpha=a)[0])
        add("spar_eg", stream_spar_eg(H_tr, y_tr, H_st, y_st, K,
                                      include_spar=False))
        add("spar_eg_plus", stream_spar_eg(H_tr, y_tr, H_st, y_st, K,
                                           include_spar=True,
                                           spar_scores=scores_adp))

        rows.append(scores)
        print(f"subject {subject}: " + " ".join(
            f"{k}={v[0]:.3f}" for k, v in scores.items()))

    print(f"\n== BCI IV {dataset} SPAR test (hybrid features, {len(rows)} subjects) ==")
    names = list(rows[0].keys())
    for name in names:
        bal = np.array([r[name][0] for r in rows])
        kap = np.array([r[name][1] for r in rows])
        print(f"{name:<16} balanced-acc {bal.mean():.3f} ± {bal.std():.3f}"
              f" | kappa {kap.mean():.3f} ± {kap.std():.3f}")

    import pandas as pd
    frame = pd.DataFrame([
        {"subject": i + 1, **{f"{k}_bal": v[0] for k, v in r.items()},
         **{f"{k}_kap": v[1] for k, v in r.items()}}
        for i, r in enumerate(rows)])
    out_csv = ROOT / "results" / f"spar_eeg_{dataset}.csv"
    frame.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    print("\n== paired tests (balanced accuracy) ==")
    for a in ("spar_full_a0.995", "spar_adp", "pam_g0.1"):
        for b in ("pam_g0.1", "leaky_a0.99", "warm"):
            x = frame[f"{a}_bal"].to_numpy()
            y = frame[f"{b}_bal"].to_numpy()
            t_stat, p_val = stats.ttest_rel(x, y)
            print(f"{a} vs {b}: diff={x.mean() - y.mean():+.4f}"
                  f" t={t_stat:.2f} p={p_val:.3f}")

    print(f"\ntotal {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
