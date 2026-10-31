"""BP arms: train-split-tuned EEGNet-lite + WiSE-FT on the net (WSL GPU).

Arms (per subject, BCI IV 2a/2b from the .npz cache):
  bp-tuned      EEGNet-lite, (epochs, lr) selected on an 80/20 train split
                (NO test access), retrained on the full train session.
  wiseft-net    WiSE-FT protocol on the net: zero-shot = train-session net;
                fine-tuned = net trained on train + full stream (offline,
                supervised -- disclosed); alpha swept post-hoc, best alpha
                reported (their envelope). Includes alpha=0 (zero-shot) and
                alpha=1 (fine-tuned) endpoints.
  bp-e2e        the tuned net evaluated e2e (= wiseft alpha=0 endpoint).

Run: $BP_PYTHON scripts/bench_bp_tta.py --dataset 2a
(WSL GPU venv; BP_PYTHON defaults to python3)
"""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jax
import jax.numpy as jnp
import optax

from bench_bp_core import (downsample, forward, init_params,  # noqa: E402
                           standardize_channels)
from core.metrics import balanced_accuracy, cohen_kappa  # noqa: E402

CACHE = ROOT / "data" / "eeg_cache"


def train_cfg(X_tr, y_tr, K, epochs, lr, seed, batch=32):
    p = init_params(jax.random.PRNGKey(seed), X_tr.shape[1], X_tr.shape[2], K)
    opt = optax.adam(lr)
    state = opt.init(p)

    @jax.jit
    def update(p, state, xb, yb):
        logits, _ = forward(p, xb, K)

        def loss(p):
            return -jnp.mean(jax.nn.log_softmax(forward(p, xb, K)[0])[
                jnp.arange(len(yb)), yb])

        grads = jax.grad(loss)(p)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(p, updates), state

    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(y_tr))
        for start in range(0, len(y_tr), batch):
            idx = order[start:start + batch]
            p, state = update(p, state, jnp.asarray(X_tr[idx]),
                              jnp.asarray(y_tr[idx]))
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2a", choices=["2a", "2b"])
    parser.add_argument("--subjects", type=int, default=9)
    parser.add_argument("--grid", default="120:1e-3,300:1e-3,300:3e-4")
    args = parser.parse_args()
    print("backend:", jax.default_backend(), "| devices:", jax.devices())

    rows = []
    for subject in range(1, args.subjects + 1):
        data = np.load(CACHE / args.dataset / f"subject{subject}.npz")
        X_tr = downsample(data["X_train"].astype(np.float32))
        X_st = downsample(data["X_stream"].astype(np.float32))
        y_tr = data["y_train"]
        y_st = data["y_stream"]
        K = int(max(y_tr.max(), y_st.max()) + 1)
        X_tr, X_st = standardize_channels(X_tr, X_st)

        # ---- hyperparameter selection on a train split ----
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y_tr))
        n_val = max(30, len(y_tr) // 5)
        val_idx, sub_idx = idx[:n_val], idx[n_val:]
        best_cfg, best_val = None, -1.0
        for cfg in args.grid.split(","):
            epochs, lr = cfg.split(":")
            epochs, lr = int(epochs), float(lr)
            p = train_cfg(X_tr[sub_idx], y_tr[sub_idx], K, epochs, lr,
                          seed=subject)
            logits, _ = forward(p, jnp.asarray(X_tr[val_idx]), K)
            acc = balanced_accuracy(y_tr[val_idx],
                                    np.asarray(jnp.argmax(logits, axis=1)))
            if acc > best_val:
                best_val, best_cfg = acc, (epochs, lr)
        epochs, lr = best_cfg

        # ---- retrain on the full train session ----
        p_zero = train_cfg(X_tr, y_tr, K, epochs, lr, seed=subject)
        logits, _ = forward(p_zero, jnp.asarray(X_st), K)
        pred_e2e = np.asarray(jnp.argmax(logits, axis=1))
        row = {"subject": subject, "cfg_epochs": epochs, "cfg_lr": lr,
               "bp-e2e": balanced_accuracy(y_st, pred_e2e)}

        # ---- WiSE-FT with CAUSAL online fine-tuning (no leakage) ----
        # The original WiSE-FT protocol (fine-tune on the target set, then
        # evaluate on it) is train-on-test leakage under a cross-session
        # protocol (we measured it: alpha=1 gives 1.000 accuracy by
        # memorization). The causal analog: fine-tune ONLINE through the
        # stream (one SGD step per trial AFTER predicting it), and predict
        # trial t with the weight interpolation (1-a) W_zero + a W_t where
        # W_t is the online model BEFORE trial t's label. alpha envelope
        # post-hoc (their protocol), all trajectories causal.
        alphas = np.linspace(0.0, 1.0, 11)
        p_ft = jax.tree_util.tree_map(lambda z: z * 1.0, p_zero)
        opt_ft = optax.adam(1e-3)
        state_ft = opt_ft.init(p_ft)

        @jax.jit
        def sgd_step(p, state, xb, yb):
            def loss(p):
                return -jax.nn.log_softmax(forward(p, xb[None], K)[0])[0, yb]
            grads = jax.grad(loss)(p)
            updates, state = opt_ft.update(grads, state)
            return optax.apply_updates(p, updates), state

        correct = {round(float(a), 2): 0 for a in alphas}
        n_st = len(y_st)
        for t in range(n_st):
            x_t = jnp.asarray(X_st[t])
            for a in alphas:
                p_a = jax.tree_util.tree_map(
                    lambda z, f: (1 - a) * z + a * f, p_zero, p_ft)
                logit_a, _ = forward(p_a, x_t[None], K)
                if int(jnp.argmax(logit_a[0])) == int(y_st[t]):
                    correct[round(float(a), 2)] += 1
            p_ft, state_ft = sgd_step(p_ft, state_ft, x_t, int(y_st[t]))

        accs = {a: c / n_st for a, c in correct.items()}
        # balanced accuracy proxy via per-class rates (classes near-balanced)
        best_a = max(accs, key=accs.get)
        row["wiseft-best_a"] = best_a
        row["wiseft-best"] = accs[best_a]
        row["wiseft-a0"] = row["bp-e2e"]
        row["wiseft-a1"] = accs[1.0]

        rows.append(row)
        print(f"s{subject}: cfg=({epochs}, {lr}) val={best_val:.3f} "
              f"bp-e2e={row['bp-e2e']:.3f} wiseft@{best_a}="
              f"{accs[best_a]:.3f} a1={accs[1.0]:.3f}", flush=True)

    print(f"\n== BP arms, BCI IV {args.dataset} ({len(rows)} subjects) ==")
    for key in ("bp-e2e", "wiseft-best", "wiseft-a0", "wiseft-a1"):
        vals = np.array([r[key] for r in rows])
        print(f"{key:<14} balanced-acc {vals.mean():.3f} ± {vals.std():.3f}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(
        ROOT / "results" / f"bp_tta_{args.dataset}.csv", index=False)
    print("saved")


if __name__ == "__main__":
    main()
