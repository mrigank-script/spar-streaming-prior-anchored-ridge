"""BP+Core upper bound: backprop-trained EEGNet-lite features + closed-form readout.

Run with the WSL GPU venv (jax + optax):
  $BP_PYTHON scripts/bench_bp_core.py --dataset 2a
(BP_PYTHON defaults to python3; see requirements-wsl.txt)
Data come from data/eeg_cache/<dataset>/subject*.npz (scripts/export_eeg_npz.py).
"""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import optax

from core.metrics import balanced_accuracy, cohen_kappa
from runner import classifier

CACHE = ROOT / "data" / "eeg_cache"


def downsample(X, factor=2):
    n, c, t = X.shape
    t2 = (t // factor) * factor
    return X[:, :, :t2].reshape(n, c, t2 // factor, factor).mean(-1)


def standardize_channels(X_tr, X_st):
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_tr - mu) / sd, (X_st - mu) / sd


def _conv(x, w, groups=1):
    return jax.lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"), feature_group_count=groups,
    )


def _avg_pool(x, k):
    return jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 1, 1, k), (1, 1, 1, k), "SAME") / k


def stem(p, x):
    h = jax.nn.elu(_conv(x[:, None], p["w1"]) + p["b1"][None, :, None, None])
    h = jax.nn.elu(_conv(h, p["w2"], groups=h.shape[1]) + p["b2"][None, :, None, None])
    h = _avg_pool(h, 4)
    h = jax.nn.elu(_conv(h, p["w3"], groups=h.shape[1]) + p["b3"][None, :, None, None])
    h = jax.nn.elu(_conv(h, p["w4"]) + p["b4"][None, :, None, None])
    h = _avg_pool(h, 8)
    return h.reshape(h.shape[0], -1)


def forward(p, x, K):
    f = jax.nn.elu(stem(p, x) @ p["w5"] + p["b5"])
    return f @ p["w6"] + p["b6"], f


def init_params(key, C, T, K, F1=8, D=2, F2=16, H=64):
    keys = jax.random.split(key, 5)
    p = {
        "w1": jax.random.normal(keys[0], (F1, 1, 1, 33)) * 0.1,
        "b1": jnp.zeros((F1,)),
        "w2": jax.random.normal(keys[1], (F1 * D, 1, C, 1)) * 0.1,
        "b2": jnp.zeros((F1 * D,)),
        "w3": jax.random.normal(keys[2], (F1 * D, 1, 1, 17)) * 0.1,
        "b3": jnp.zeros((F1 * D,)),
        "w4": jax.random.normal(keys[3], (F2, F1 * D, 1, 1)) * 0.1,
        "b4": jnp.zeros((F2,)),
    }
    fdim = stem(p, jnp.zeros((1, C, T))).shape[-1]
    keys2 = jax.random.split(keys[4], 2)
    p["w5"] = jax.random.normal(keys2[0], (fdim, H)) * jnp.sqrt(2.0 / fdim)
    p["b5"] = jnp.zeros((H,))
    p["w6"] = jax.random.normal(keys2[1], (H, K)) * jnp.sqrt(2.0 / H)
    p["b6"] = jnp.zeros((K,))
    return p


def loss_fn(p, xb, yb, K):
    logits, _ = forward(p, xb, K)
    return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(len(yb)), yb])


def train(X_tr, y_tr, K, epochs, seed, batch=32):
    C, T = X_tr.shape[1], X_tr.shape[2]
    p = init_params(jax.random.PRNGKey(seed), C, T, K)
    opt = optax.adam(1e-3)
    state = opt.init(p)

    @jax.jit
    def update(p, state, xb, yb):
        grads = jax.grad(loss_fn)(p, xb, yb, K)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(p, updates), state

    embed = jax.jit(lambda p, x: forward(p, x, K)[1])
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(y_tr))
        for start in range(0, len(y_tr), batch):
            idx = order[start:start + batch]
            p, state = update(p, state, jnp.asarray(X_tr[idx]), jnp.asarray(y_tr[idx]))
    return p, embed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2a")
    parser.add_argument("--subjects", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=120)
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

        p, embed = train(X_tr, y_tr, K, args.epochs, seed=subject)
        logits, _ = forward(p, jnp.asarray(X_st), K)
        pred_e2e = np.asarray(jnp.argmax(logits, axis=1))

        f_tr = np.asarray(embed(p, jnp.asarray(X_tr)))
        f_st = np.asarray(embed(p, jnp.asarray(X_st)))
        mu = f_tr.mean(0)
        sd = f_tr.std(0) + 1e-8
        z_tr = (f_tr - mu) / sd
        z_st = (f_st - mu) / sd

        W = classifier.fit_warm(z_tr, y_tr, K, ridge=0.01, gamma=0.0)
        pred_warm = np.argmax(z_st @ W, axis=1)
        pred_pam = classifier.stream(z_tr, y_tr, z_st, y_st, K, alpha=1.0, gamma=0.1,
                                     G=W, reanchor_every=5)

        row = {
            "subject": subject,
            "bp-e2e": balanced_accuracy(y_st, pred_e2e),
            "bp-core-warm": balanced_accuracy(y_st, pred_warm),
            "bp-core-pam": balanced_accuracy(y_st, pred_pam),
            "kappa": cohen_kappa(y_st, pred_pam),
        }
        rows.append(row)
        print(f"s{subject}: bp-e2e={row['bp-e2e']:.3f} bp+core-warm={row['bp-core-warm']:.3f} "
              f"bp+core-pam={row['bp-core-pam']:.3f} kappa(pam)={row['kappa']:.3f}")

    print(f"\n== BP+Core, BCI IV {args.dataset} ({len(rows)} subjects, {args.epochs} epochs) ==")
    for key in ("bp-e2e", "bp-core-warm", "bp-core-pam"):
        vals = np.array([r[key] for r in rows])
        print(f"{key:<14} balanced-acc {vals.mean():.3f} ± {vals.std():.3f}")


if __name__ == "__main__":
    main()
