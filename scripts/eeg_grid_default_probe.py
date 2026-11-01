"""Lightweight probe: one EEG arm (hybrid + SPAR-Pull head) at the factorial
grid defaults, to price the single stream-dependent deviation of Table E.1.

Runs only the hybrid feature pipeline and the one head, so it is far cheaper
than the full tier-1 suite.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
from core import features  # noqa: E402
from dataio import load_eeg_subject  # noqa: E402
from dataio.eeg import CUE_NAMES  # noqa: E402
from bench_eeg_tier1 import (FEAT_CFG, bandpass, cov_features,  # noqa: E402
                             ea_whitener, feat_eval, stream_spar_kf_eeg)
from core.metrics import balanced_accuracy, cohen_kappa  # noqa: E402

GRID = dict(refresh_every=10, resid_fast=30, w_min=5e-4, w_max=0.05,
            K_cal=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2a", choices=["2a", "2b"])
    ap.add_argument("--subjects", type=int, default=9)
    args = ap.parse_args()

    feature_fn = features.make_feature_fn(FEAT_CFG)
    bal, kap = [], []
    t0 = time.perf_counter()
    for subject in range(1, args.subjects + 1):
        out = load_eeg_subject(dataset=args.dataset, subject=subject)
        (X_tr, y_tr), (X_st, y_st) = out["train"], out["stream"]
        K = len(CUE_NAMES[args.dataset])
        W_ea_tr = ea_whitener(X_tr)
        W_ea_st = ea_whitener(X_st)
        phi_tr, phi_st = feat_eval(
            np.einsum("nct,cd->ndt", X_tr, W_ea_tr),
            np.einsum("nct,cd->ndt", X_st, W_ea_st), feature_fn)
        Xbp_tr = bandpass(np.einsum("nct,cd->ndt", X_tr, W_ea_tr))
        Xbp_st = bandpass(np.einsum("nct,cd->ndt", X_st, W_ea_st))
        ref2 = cov_features(Xbp_tr).mean(0)
        Gc_tr = cov_features(Xbp_tr, ref2)
        Gc_st = cov_features(Xbp_st, ref2)
        H_tr = np.concatenate(
            [phi_tr, (Gc_tr - Gc_tr.mean(0)) / (Gc_tr.std(0) + 1e-12)], axis=1)
        H_st = np.concatenate(
            [phi_st, (Gc_st - Gc_tr.mean(0)) / (Gc_tr.std(0) + 1e-12)], axis=1)
        preds = stream_spar_kf_eeg(H_tr, y_tr, H_st, y_st, K, **GRID)[0]
        b = balanced_accuracy(y_st, preds)
        bal.append(b)
        kap.append(cohen_kappa(y_st, preds))
        print(f"subject {subject}: bal={b:.4f} ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    bal = np.array(bal)
    kap = np.array(kap)
    print(f"\n[{args.dataset}] hybrid + SPAR-Pull (grid defaults): "
          f"bal {bal.mean():.4f} ± {bal.std(ddof=1):.4f} | "
          f"kappa {kap.mean():.4f} ± {kap.std(ddof=1):.4f}")
    np.save(ROOT / "results" / f"eeg_griddefault_{args.dataset}.npy", bal)
    import pandas as pd
    pd.DataFrame(dict(subject=np.arange(1, len(bal) + 1), bal=bal,
                      kap=kap)).to_csv(
        ROOT / "results" / f"eeg_griddefault_{args.dataset}.csv", index=False)


if __name__ == "__main__":
    main()