"""EEG chapter pipeline: cross-session BCI IV 2a/2b with closed-form heads."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import features
from core.metrics import balanced_accuracy, cohen_kappa
from dataio import load_eeg_subject
from dataio.eeg import CUE_NAMES
from runner import classifier

FEAT_CFG = features.FeatureConfig(stats=True, lags=(1, 2, 4, 8), fft_bins=4, lru_modes=0)
DOWNSAMPLE = 20


def downsample(trials):
    n, c, t = trials.shape
    bins = t // DOWNSAMPLE
    return trials[:, :, :bins * DOWNSAMPLE].reshape(n, c, bins, DOWNSAMPLE).mean(-1).reshape(n, -1)


def ea_whitener(trials):
    n, c, t = trials.shape
    flat = trials.transpose(1, 0, 2).reshape(c, -1)
    cov = flat @ flat.T / flat.shape[1]
    cov = cov + 1e-6 * (np.trace(cov) / c) * np.eye(c)
    vals, vecs = np.linalg.eigh(cov)
    return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T


def apply_whitener(trials, W):
    return np.einsum("nct,cd->ndt", trials, W)


def bandpass(trials, low=8.0, high=30.0, fs=250.0, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, trials, axis=-1)


def cov_features(trials, ref=None):
    """Per-trial log-Euclidean covariance tangent features (lower triangle)."""
    n, c, t = trials.shape
    tril = np.tril_indices(c)
    out = np.zeros((n, len(tril[0])))
    for i in range(n):
        cov = trials[i] @ trials[i].T / t
        cov = cov + 1e-8 * (np.trace(cov) / c) * np.eye(c)
        vals, vecs = np.linalg.eigh(cov)
        logm = (vecs * np.log(np.maximum(vals, 1e-30))) @ vecs.T
        out[i] = logm[tril]
    return out if ref is None else out - ref


def lda_eval(X_tr, y_tr, X_st, y_st):
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_st)
    return balanced_accuracy(y_st, pred), cohen_kappa(y_st, pred)


def feat_eval(trials_tr, trials_st, feature_fn):
    win_tr = jnp.asarray(trials_tr.transpose(0, 2, 1))
    win_st = jnp.asarray(trials_st.transpose(0, 2, 1))
    phi_tr = np.asarray(jax.vmap(feature_fn)(win_tr).block_until_ready())
    phi_st = np.asarray(jax.vmap(feature_fn)(win_st).block_until_ready())
    mu = phi_tr.mean(0)
    sd = phi_tr.std(0) + 1e-8
    return (phi_tr - mu) / sd, (phi_st - mu) / sd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2a", choices=["2a", "2b"])
    parser.add_argument("--subjects", type=int, default=9)
    args = parser.parse_args()
    print("backend:", jax.default_backend(), "| x64:", jax.config.jax_enable_x64)

    feature_fn = features.make_feature_fn(FEAT_CFG)
    rows = []
    t_start = time.perf_counter()
    for subject in range(1, args.subjects + 1):
        out = load_eeg_subject(dataset=args.dataset, subject=subject)
        (X_tr, y_tr), (X_st, y_st) = out["train"], out["stream"]
        K = len(CUE_NAMES[args.dataset])
        scores = {}

        scores["lda-raw"] = lda_eval(downsample(X_tr), y_tr, downsample(X_st), y_st)
        W_ea_tr = ea_whitener(X_tr)
        W_ea_st = ea_whitener(X_st)
        scores["lda-raw-EA"] = lda_eval(
            downsample(apply_whitener(X_tr, W_ea_tr)), y_tr,
            downsample(apply_whitener(X_st, W_ea_st)), y_st,
        )

        phi_tr, phi_st = feat_eval(X_tr, X_st, feature_fn)
        scores["lda-feat"] = lda_eval(phi_tr, y_tr, phi_st, y_st)

        phi_tr_ea, phi_st_ea = feat_eval(
            apply_whitener(X_tr, W_ea_tr), apply_whitener(X_st, W_ea_st), feature_fn
        )
        scores["lda-feat-EA"] = lda_eval(phi_tr_ea, y_tr, phi_st_ea, y_st)

        W_warm = classifier.fit_warm(phi_tr, y_tr, K, ridge=0.01, gamma=0.0)
        pred_warm = np.argmax(phi_st @ W_warm, axis=1)
        scores["aal-warm"] = (balanced_accuracy(y_st, pred_warm), cohen_kappa(y_st, pred_warm))

        pred_pam = classifier.stream(phi_tr, y_tr, phi_st, y_st, K, alpha=1.0, gamma=0.1,
                                     G=W_warm, reanchor_every=5)
        scores["aal-pam"] = (balanced_accuracy(y_st, pred_pam), cohen_kappa(y_st, pred_pam))

        W_warm_ea = classifier.fit_warm(phi_tr_ea, y_tr, K, ridge=0.01, gamma=0.0)
        pred_warm_ea = np.argmax(phi_st_ea @ W_warm_ea, axis=1)
        scores["ea-aal-warm"] = (balanced_accuracy(y_st, pred_warm_ea), cohen_kappa(y_st, pred_warm_ea))

        pred_pam_ea = classifier.stream(phi_tr_ea, y_tr, phi_st_ea, y_st, K, alpha=1.0, gamma=0.1,
                                        G=W_warm_ea, reanchor_every=5)
        scores["ea-aal-pam"] = (balanced_accuracy(y_st, pred_pam_ea), cohen_kappa(y_st, pred_pam_ea))

        Xbp_tr = bandpass(apply_whitener(X_tr, W_ea_tr))
        Xbp_st = bandpass(apply_whitener(X_st, W_ea_st))
        ref_cov = cov_features(Xbp_tr).mean(0)
        Fc_tr = cov_features(Xbp_tr, ref_cov)
        Fc_st = cov_features(Xbp_st, ref_cov)
        scores["ea-cov-lda"] = lda_eval(Fc_tr, y_tr, Fc_st, y_st)
        mc = Fc_tr.mean(0)
        sc = Fc_tr.std(0) + 1e-12
        Fz_tr = (Fc_tr - mc) / sc
        Fz_st = (Fc_st - mc) / sc
        W_cov = classifier.fit_warm(Fz_tr, y_tr, K, ridge=0.01, gamma=0.0)
        pred_cov = np.argmax(Fz_st @ W_cov, axis=1)
        scores["ea-cov-warm"] = (balanced_accuracy(y_st, pred_cov), cohen_kappa(y_st, pred_cov))
        pred_cov_pam = classifier.stream(Fz_tr, y_tr, Fz_st, y_st, K, alpha=1.0, gamma=0.1,
                                         G=W_cov, reanchor_every=5)
        scores["ea-cov-pam"] = (balanced_accuracy(y_st, pred_cov_pam), cohen_kappa(y_st, pred_cov_pam))

        H_tr = np.concatenate([phi_tr_ea, Fz_tr], axis=1)
        H_st = np.concatenate([phi_st_ea, Fz_st], axis=1)
        W_h = classifier.fit_warm(H_tr, y_tr, K, ridge=0.01, gamma=0.0)
        pred_h = np.argmax(H_st @ W_h, axis=1)
        scores["ea-hybrid-warm"] = (balanced_accuracy(y_st, pred_h), cohen_kappa(y_st, pred_h))
        pred_h_pam = classifier.stream(H_tr, y_tr, H_st, y_st, K, alpha=1.0, gamma=0.1,
                                       G=W_h, reanchor_every=5)
        scores["ea-hybrid-pam"] = (balanced_accuracy(y_st, pred_h_pam), cohen_kappa(y_st, pred_h_pam))

        lda_cov = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Fc_tr, y_tr)
        p_cov = lda_cov.predict_proba(Fc_st)
        logits = H_st @ W_h
        z = (logits - logits.mean(axis=1, keepdims=True)) / (logits.std(axis=1, keepdims=True) + 1e-9)
        p_hyb = np.exp(z)
        p_hyb /= p_hyb.sum(axis=1, keepdims=True)
        ens = np.argmax(0.5 * p_cov + 0.5 * p_hyb, axis=1)
        scores["ens-covlda+hybrid"] = (balanced_accuracy(y_st, ens), cohen_kappa(y_st, ens))

        rows.append(scores)
        summary = " ".join(f"{name}={scores[name][0]:.3f}" for name in scores)
        print(f"subject {subject}: {summary}")

    print(f"\n== BCI IV {args.dataset} cross-session (mean over {len(rows)} subjects) ==")
    for name in rows[0]:
        bal = np.array([r[name][0] for r in rows])
        kap = np.array([r[name][1] for r in rows])
        print(f"{name:<12} balanced-acc {bal.mean():.3f} ± {bal.std():.3f} | kappa {kap.mean():.3f} ± {kap.std():.3f}")

    import pandas as pd
    from scipy import stats

    frame = pd.DataFrame([
        {"subject": i + 1,
         **{f"{name}_bal": v[0] for name, v in scores.items()},
         **{f"{name}_kappa": v[1] for name, v in scores.items()}}
        for i, scores in enumerate(rows)
    ])
    out_csv = ROOT / "results" / f"eeg_{args.dataset}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    for a, b in (("ea-hybrid-pam", "lda-raw-EA"), ("ea-hybrid-pam", "ea-hybrid-warm"),
                 ("ea-cov-pam", "ea-cov-warm"), ("aal-pam", "aal-warm"),
                 ("ens-covlda+hybrid", "ea-cov-lda"), ("ens-covlda+hybrid", "ea-hybrid-pam")):
        if f"{a}_bal" in frame and f"{b}_bal" in frame:
            x = frame[f"{a}_bal"].to_numpy()
            y = frame[f"{b}_bal"].to_numpy()
            t_stat, p_val = stats.ttest_rel(x, y)
            try:
                _, p_wil = stats.wilcoxon(x, y)
            except Exception:
                p_wil = float("nan")
            print(f"{a} vs {b}: diff={x.mean() - y.mean():+.3f} t={t_stat:.2f} p={p_val:.3f} wilcoxon_p={p_wil:.3f}")
    print(f"total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
