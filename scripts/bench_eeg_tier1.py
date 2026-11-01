"""Tier-1 EEG benchmark — classical baselines + the self-calibrating SPAR-Pull head.

Arms (BCI IV 2a + 2b, cached epochs, cross-session protocol):
  Classical baselines (published, canonical implementations):
    mdrm            pyriemann MDM (Riemannian minimum-distance-to-mean)
    fgmdm           pyriemann FgMDM (a standard MI-BCI baseline)
    csp-lda         mne CSP(6) + shrinkage LDA (the MI classical standard)
    tscov-lda       log-Euclidean tangent cov features + shrinkage LDA
  Ours (closed-form streaming, supervised predict-then-update):
    csp-spar_kf     CSP features + self-calibrating head (anchor rho* + window w*)
    tancov-spar_kf  tangent cov features + self-calibrating head
    hybrid-spar_kf  hybrid features + self-calibrating head
    hybrid-pam_g1   the previous best tuned anchored arm (reference)
    hybrid-leaky    forgetting-only streaming reference
    olea-hybrid-spar_kf  causal online-EA alignment + hybrid + self-calibrating head
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

from bench_eeg import bandpass, cov_features, ea_whitener, feat_eval  # noqa: E402
from core.metrics import balanced_accuracy, cohen_kappa  # noqa: E402
from dataio import load_eeg_subject  # noqa: E402
from dataio.eeg import CUE_NAMES  # noqa: E402
from runner import classifier  # noqa: E402

from core import features  # noqa: E402
from bench_eeg import FEAT_CFG  # noqa: E402

from pyriemann.classification import FgMDM, MDM  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from mne.decoding import CSP  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402


def onehot(y, K):
    Y = np.zeros((len(y), K), dtype=np.float64)
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return Y


def stream_spar_kf_eeg(X_tr, y_tr, X_st, y_st, K, refresh_every=5,
                       resid_fast=40, w_min=5e-3, w_max=0.02, K_cal=2,
                       eta_C=0.2, eta_nu=0.1, clip_hi=1.0, rho_init=0.1,
                       gamma=1.0, return_diag=False):
    """Self-calibrating streaming head (multi-class): anchor strength rho* (conflict
    theorem) + forgetting window w* (window sign theorem), both from the
    two-scale residual statistics. Adapted from run_spar_kf for short
    EEG streams (B is D x K; short-stream horizons)."""
    D = X_tr.shape[1]
    A = X_tr.T @ X_tr.astype(np.float64)
    B = X_tr.T @ onehot(y_tr, K)
    n_eff = float(len(X_tr))

    W_warm_fit = np.linalg.solve(A + n_eff * 0.01 * np.eye(D), B)
    G = W_warm_fit.copy()
    G_ref = G.copy()
    resid_w = onehot(y_tr, K) - X_tr @ W_warm_fit
    sigma2_base = float(np.var(resid_w)) + 1e-12
    sigma2_fast = sigma2_base

    eta_f = 1.0 / resid_fast
    eta_ref = refresh_every / 150.0
    C_ema = None
    nu2_ema = 0.0
    w = w_min
    alpha = 1.0 - w
    rho = rho_init

    Yst = onehot(y_st, K)
    preds = np.zeros(len(y_st), dtype=int)
    scores_out = np.zeros((len(y_st), K))
    rhos, ws = [], []
    step = 0
    cyc = 0

    for i in range(len(y_st)):
        x = X_st[i]
        if step % refresh_every == 0:
            Abar = A / n_eff
            mu_bar = float(np.trace(Abar) / D) + 1e-12
            mu_all = np.linalg.eigvalsh(Abar)
            d_act = max(1, int(np.sum(
                mu_all > 1e-8 * max(float(mu_all.max()), 1e-12))))
            V_hat = sigma2_base * d_act * K / n_eff
            W_data = np.linalg.solve(A + n_eff * 1e-6 * np.eye(D), B)
            delta = G_ref - W_data
            C_ref = float(np.sum(delta * (Abar @ delta))) - V_hat
            E_resid = max(0.0, sigma2_fast - sigma2_base)
            C_resid = E_resid
            C_floor = 0.5 * sigma2_base * d_act * K / n_eff
            if C_ema is None:
                C_ema = 2.0 * C_floor
            C_ema = (1 - eta_C) * C_ema + eta_C * max(C_ref, C_resid,
                                                      C_floor, 1e-10)
            rho_raw = mu_bar * V_hat / C_ema
            cap = rho_init if cyc < K_cal else clip_hi
            rho = float(np.clip(rho_raw, 1e-5, cap))

            # window estimator: forgetting rate (variance-floor-subtracted inversion)
            s_bar = rho / (mu_bar + rho)
            c_prime = (1.0 - s_bar) ** 2 * sigma2_base * D
            nu2_hat = max(E_resid * w * w
                          - c_prime * w ** 3 / max(2.0 - w, 1e-6), 0.0)
            nu2_ema = (1 - eta_nu) * nu2_ema + eta_nu * nu2_hat
            kappa = 4.0 * nu2_ema / max(c_prime, 1e-12)
            w_star = kappa ** (1.0 / 3.0)
            w = w_min if cyc < K_cal else float(
                np.clip(w_star, w_min, w_max))
            alpha = 1.0 - w

        W = np.linalg.solve(A + n_eff * rho * np.eye(D),
                            B + n_eff * rho * gamma * G)
        score = x @ W
        scores_out[i] = score
        preds[i] = int(np.argmax(score))
        rhos.append(rho)
        ws.append(w)

        e_t = float(np.mean((Yst[i] - score) ** 2))
        sigma2_fast = (1 - eta_f) * sigma2_fast + eta_f * e_t
        sigma2_base = (1 - 5e-4) * sigma2_base + 5e-4 * e_t

        if step % refresh_every == 0 and step > 0:
            Abar = A / n_eff
            mu, U = np.linalg.eigh(Abar)
            beta = (mu / (mu + rho))[:, None]
            Gp = U.T @ G
            Wp = U.T @ W
            G = U @ (beta * Wp + (1 - beta) * Gp)
            G_ref = (1 - eta_ref) * G_ref + eta_ref * W
            cyc += 1

        A = alpha * A + np.outer(x, x)
        B = alpha * B + np.outer(x, Yst[i])
        n_eff = alpha * n_eff + 1.0
        step += 1

    if return_diag:
        return preds, np.array(rhos), np.array(ws), scores_out
    return preds, scores_out


def leaky_scores_generic(X_tr, y_tr, X_st, y_st, K, alpha=1.0, ridge=0.01):
    """Causal leaky-ridge score path for the Hedge wrapper."""
    D = X_tr.shape[1]
    A = X_tr.T @ X_tr.astype(np.float64)
    B = X_tr.T @ onehot(y_tr, K)
    n = float(len(X_tr))
    Yst = onehot(y_st, K)
    out = np.zeros((len(y_st), K))
    for i, x in enumerate(X_st):
        W = np.linalg.solve(A + n * ridge * np.eye(D), B)
        out[i] = x @ W
        A = alpha * A + np.outer(x, x)
        B = alpha * B + np.outer(x, Yst[i])
        n = alpha * n + 1.0
    return out


def hedge_scores(score_list, y_st, K):
    """Causal exponentially weighted mixture over score paths."""
    Kexp = len(score_list)
    pi = np.ones(Kexp) / Kexp
    Yst = onehot(y_st, K)
    preds = np.zeros(len(y_st), dtype=int)
    S = 1.0
    for i in range(len(y_st)):
        sc = np.stack([s[i] for s in score_list])
        mix = pi @ sc
        preds[i] = int(np.argmax(mix))
        losses = np.mean((sc - Yst[i]) ** 2, axis=1)
        eta = min(0.5 / max(float(losses.max()), 1e-6),
                  np.sqrt(np.log(Kexp) / max(S, 1.0)))
        pi = pi * np.exp(-eta * losses)
        pi /= pi.sum()
        S += float(np.mean((mix - Yst[i]) ** 2))
    return preds


def online_ea_features(X_tr, X_st, feature_fn, window=40, every=10):
    """Causal (online) Euclidean Alignment: whitener from the trailing
    `window` most recent trials, refreshed every `every` stream trials.
    Unsupervised (uses only stream x). Train features use the offline
    train whitener so the warm-up matches the other arms."""
    # offline train whitener (train session only)
    flat = X_tr.transpose(1, 0, 2).reshape(X_tr.shape[1], -1)
    cov = flat @ flat.T / flat.shape[1]
    cov = cov + 1e-6 * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
    vals, vecs = np.linalg.eigh(cov)
    W_tr = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
    import jax
    phi_tr = np.asarray(jax.vmap(feature_fn)(
        np.einsum("nct,cd->ndt", X_tr, W_tr).transpose(0, 2, 1)))

    hist = [X_tr[i] for i in range(len(X_tr) - window, len(X_tr))]
    feats_st = np.zeros((len(X_st), phi_tr.shape[1]))
    Wh = W_tr
    import jax
    feat_one = jax.jit(feature_fn)
    for i in range(len(X_st)):
        if i % every == 0:
            win = np.stack(hist[-window:])
            flat = win.transpose(1, 0, 2).reshape(win.shape[1], -1)
            cov = flat @ flat.T / flat.shape[1]
            cov = cov + 1e-6 * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
            vals, vecs = np.linalg.eigh(cov)
            Wh = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
        feats_st[i] = np.asarray(feat_one((Wh @ X_st[i]).T))
        hist.append(X_st[i])
        if len(hist) > window + 400:
            hist.pop(0)
    mu = phi_tr.mean(0)
    sd = phi_tr.std(0) + 1e-8
    return (phi_tr - mu) / sd, (feats_st - mu) / sd


def online_ea_tangent_stream(X_tr, X_st, ref_cov, window=40, every=10):
    """Causal tangent-covariance features: the whitener of each stream block
    comes from the trailing `window` most recent trials (refreshed every
    `every` trials), so stream trial i never sees trials > i. The train-side
    reference `ref_cov` is computed from the training session only."""
    hist = [X_tr[i] for i in range(len(X_tr) - window, len(X_tr))]
    feats = None
    for start in range(0, len(X_st), every):
        stop = min(start + every, len(X_st))
        win = np.stack(hist[-window:])
        flat = win.transpose(1, 0, 2).reshape(win.shape[1], -1)
        cov = flat @ flat.T / flat.shape[1]
        cov = cov + 1e-6 * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
        vals, vecs = np.linalg.eigh(cov)
        Wh = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
        Gc_blk = cov_features(
            bandpass(np.einsum("nct,cd->ndt", X_st[start:stop], Wh)), ref_cov)
        if feats is None:
            feats = np.zeros((len(X_st),) + Gc_blk.shape[1:])
        feats[start:stop] = Gc_blk
        for i in range(start, stop):
            hist.append(X_st[i])
            if len(hist) > window + 400:
                hist.pop(0)
    return feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2a", choices=["2a", "2b"])
    parser.add_argument("--subjects", type=int, default=9)
    args = parser.parse_args()

    import jax
    feature_fn = features.make_feature_fn(FEAT_CFG)
    rows = []
    t_start = time.perf_counter()
    for subject in range(1, args.subjects + 1):
        out = load_eeg_subject(dataset=args.dataset, subject=subject)
        (X_tr, y_tr), (X_st, y_st) = out["train"], out["stream"]
        K = len(CUE_NAMES[args.dataset])
        scores = {}

        def add(name, preds):
            scores[name] = (balanced_accuracy(y_st, preds),
                            cohen_kappa(y_st, preds))

        # --- classical baselines on bandpassed epochs ---
        Xb_tr = bandpass(X_tr)
        Xb_st = bandpass(X_st)
        cov_tr = Covariances("lwf").fit_transform(Xb_tr)
        cov_st = Covariances("lwf").fit_transform(Xb_st)
        mdrm = MDM(metric="riemann").fit(cov_tr, y_tr)
        add("mdrm", mdrm.predict(cov_st))
        fgmdm = FgMDM().fit(cov_tr, y_tr)
        add("fgmdm", fgmdm.predict(cov_st))

        csp = CSP(n_components=6, reg="ledoit_wolf", log=True)
        F_tr = csp.fit_transform(Xb_tr, y_tr)
        F_st = csp.transform(Xb_st)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        add("csp-lda", lda.fit(F_tr, y_tr).predict(F_st))

        # tangent cov + LDA (TSMDM-style)
        ref_cov = cov_features(Xb_tr).mean(0)
        Fc_tr = cov_features(Xb_tr, ref_cov)
        Fc_st = cov_features(Xb_st, ref_cov)
        lda2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        add("tscov-lda", lda2.fit(Fc_tr, y_tr).predict(Fc_st))

        # --- ours: streaming heads ---
        add("csp-spar_kf", stream_spar_kf_eeg(F_tr, y_tr, F_st, y_st, K)[0])
        Fz_tr = (Fc_tr - Fc_tr.mean(0)) / (Fc_tr.std(0) + 1e-12)
        Fz_st = (Fc_st - Fc_tr.mean(0)) / (Fc_tr.std(0) + 1e-12)
        add("tancov-spar_kf", stream_spar_kf_eeg(Fz_tr, y_tr, Fz_st, y_st, K)[0])

        # hybrid features (previous best pipeline)
        W_ea_tr = ea_whitener(X_tr)
        W_ea_st = ea_whitener(X_st)
        phi_tr_ea, phi_st_ea = feat_eval(
            np.einsum("nct,cd->ndt", X_tr, W_ea_tr),
            np.einsum("nct,cd->ndt", X_st, W_ea_st), feature_fn)
        Xbp_tr = bandpass(np.einsum("nct,cd->ndt", X_tr, W_ea_tr))
        Xbp_st = bandpass(np.einsum("nct,cd->ndt", X_st, W_ea_st))
        ref2 = cov_features(Xbp_tr).mean(0)
        Gc_tr = cov_features(Xbp_tr, ref2)
        Gc_st = cov_features(Xbp_st, ref2)
        H_tr = np.concatenate([phi_tr_ea, (Gc_tr - Gc_tr.mean(0)) / (Gc_tr.std(0) + 1e-12)], axis=1)
        H_st = np.concatenate([phi_st_ea, (Gc_st - Gc_tr.mean(0)) / (Gc_tr.std(0) + 1e-12)], axis=1)
        pred_h, scores_h = stream_spar_kf_eeg(H_tr, y_tr, H_st, y_st, K)
        add("hybrid-spar_kf", pred_h)
        # probe: the same head with the *factorial-grid* control parameters
        # (no EEG-specific window range or cadence); reported as a
        # sensitivity row so the cost of the one stream-dependent deviation
        # is visible.
        add("hybrid-spar_kf_grid", stream_spar_kf_eeg(
            H_tr, y_tr, H_st, y_st, K, refresh_every=10, resid_fast=30,
            w_min=5e-4, w_max=0.05, K_cal=10)[0])

        W_h = classifier.fit_warm(H_tr, y_tr, K, ridge=0.01, gamma=0.0)
        add("hybrid-pam_g1", classifier.stream(
            H_tr, y_tr, H_st, y_st, K, alpha=1.0, gamma=1.0, G=W_h,
            reanchor_every=5))
        add("hybrid-leaky", classifier.stream(
            H_tr, y_tr, H_st, y_st, K, alpha=0.99, gamma=0.0, reanchor_every=5))

        # Hedge composition: leaky windows + frozen anchor (+ SPAR-Pull expert)
        experts = [leaky_scores_generic(H_tr, y_tr, H_st, y_st, K, alpha=a)
                   for a in (1.0, 0.995, 0.99, 0.98)]
        experts.append(H_st @ W_h)
        add("hybrid-spar_eg", hedge_scores(experts, y_st, K))
        experts.append(scores_h)
        add("hybrid-spar_eg_plus", hedge_scores(experts, y_st, K))

        # online-EA + hybrid + spar_kf (causal alignment, both feature halves)
        try:
            oe_tr, oe_st = online_ea_features(X_tr, X_st, feature_fn)
            Gc_oe_st = online_ea_tangent_stream(X_tr, X_st, ref2)
            OH_tr = np.concatenate([oe_tr, (Gc_tr - Gc_tr.mean(0)) / (Gc_tr.std(0) + 1e-12)], axis=1)
            OH_st = np.concatenate([oe_st, (Gc_oe_st - Gc_tr.mean(0)) / (Gc_tr.std(0) + 1e-12)], axis=1)
            add("olea-hybrid-spar_kf", stream_spar_kf_eeg(
                OH_tr, y_tr, OH_st, y_st, K)[0])
        except Exception as exc:  # noqa: BLE001
            print(f"  online-EA arm failed on subject {subject}: {exc}")
            scores["olea-hybrid-spar_kf"] = (float("nan"), float("nan"))

        rows.append(scores)
        print(f"subject {subject}: " + " ".join(
            f"{k}={v[0]:.3f}" for k, v in scores.items()), flush=True)

    print(f"\n== BCI IV {args.dataset} Tier-1 arms ({len(rows)} subjects) ==")
    names = list(rows[0].keys())
    for name in names:
        bal = np.array([r[name][0] for r in rows])
        kap = np.array([r[name][1] for r in rows])
        print(f"{name:<20} balanced-acc {bal.mean():.3f} ± {bal.std():.3f}"
              f" | kappa {kap.mean():.3f} ± {kap.std():.3f}")

    import pandas as pd
    frame = pd.DataFrame([
        {"subject": i + 1, **{f"{k}_bal": v[0] for k, v in r.items()},
         **{f"{k}_kap": v[1] for k, v in r.items()}}
        for i, r in enumerate(rows)])
    out_csv = ROOT / "results" / f"eeg_tier1_{args.dataset}.csv"
    frame.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    # Holm-adjusted paired tests vs the best classical arm
    from scipy import stats
    def holm(pvals):
        order = np.argsort(pvals)
        m = len(pvals)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * pvals[idx])
            adj[idx] = min(1.0, running)
        return adj

    classical = [n for n in names if n in ("mdrm", "fgmdm", "csp-lda", "tscov-lda")]
    ours = [n for n in names if "spar" in n]
    best_classical = max(classical, key=lambda n: np.mean([r[n][0] for r in rows]))
    print(f"\nbest classical arm: {best_classical}")
    pvals, pairs = [], []
    for oname in ours:
        x = np.array([r[oname][0] for r in rows])
        yc = np.array([r[best_classical][0] for r in rows])
        t_stat, p_val = stats.ttest_rel(x, yc)
        d = (x - yc).mean() / (x - yc).std() + 1e-12
        pvals.append(p_val)
        pairs.append((oname, x.mean() - yc.mean(), t_stat, d))
    adj = holm(np.array(pvals))
    print("Holm-adjusted paired tests (ours vs best classical):")
    for (oname, diff, t_stat, d), p_adj in zip(pairs, adj):
        print(f"  {oname:<20} diff={diff:+.4f} t={t_stat:.2f}"
              f" Cohen's d={d:+.2f} p_holm={p_adj:.3f}")

    print(f"total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
