"""Generate the synthetic drifting-teacher suite into data/synthetic/.

Teacher: y_t = x_t . w_t + noise, with w_t = w0 + sum_k c_k(t) u_k,
AR(1) inputs x_t, and four drift types: abrupt, gradual, recurring, rotating.
Saved per file: x, y, c, U, w0, drift onsets, meta (npz).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"

DIMS = (32, 64, 256)
DRIFTS = ("abrupt", "gradual", "recurring", "rotating")
K = 4
RHO = 0.9
X_NOISE = 1.0
Y_NOISE = 0.1


def seeds_for(length: int, dim: int) -> int:
    if length == 10_000:
        return 10 if dim < 256 else 5
    return 3 if dim < 256 else 0


def orthonormal_rows(rng: np.random.Generator, dim: int, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
    return q.T.astype(np.float32)


def drift_path(kind: str, length: int, rng: np.random.Generator, n_onsets: int = 4):
    c = np.zeros((length, K), dtype=np.float32)
    onsets = np.array([], dtype=np.int64)
    if kind == "abrupt":
        onsets = np.sort(rng.choice(np.arange(length // 10, length - length // 10), size=n_onsets, replace=False))
        levels = rng.normal(size=(n_onsets + 1, K)).astype(np.float32)
        c[:onsets[0]] = levels[0]
        for i, onset in enumerate(onsets):
            end = onsets[i + 1] if i + 1 < len(onsets) else length
            c[onset:end] = levels[i + 1]
    elif kind == "gradual":
        onsets = np.sort(rng.choice(np.arange(length // 10, length - length // 10), size=n_onsets, replace=False))
        anchors = np.vstack([np.zeros((1, K), dtype=np.float32), np.cumsum(rng.normal(size=(n_onsets + 1, K)), axis=0)]).astype(np.float32)
        knots = np.concatenate([[0], onsets, [length]])
        for i in range(len(knots) - 1):
            lo, hi = int(knots[i]), int(knots[i + 1])
            t = np.linspace(0.0, 1.0, hi - lo, endpoint=False, dtype=np.float32)[:, None]
            c[lo:hi] = (1.0 - t) * anchors[i] + t * anchors[i + 1]
    elif kind == "recurring":
        index = np.arange(length)
        for k in range(K):
            period = rng.uniform(length / 8, length / 2)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            c[:, k] = (rng.normal() * np.sin(2.0 * np.pi * index / period + phase)).astype(np.float32)
    elif kind == "rotating":
        omega = 2.0 * np.pi * rng.uniform(1.0, 3.0) / length
        index = np.arange(length)
        amp = rng.normal(size=2)
        c[:, 0] = amp[0] * np.cos(omega * index)
        c[:, 1] = amp[0] * np.sin(omega * index)
        c[:, 2] = amp[1] * np.cos(1.7 * omega * index)
        c[:, 3] = amp[1] * np.sin(1.7 * omega * index)
    else:
        raise ValueError(f"unknown drift kind: {kind}")
    return c, onsets


def generate_config(dim: int, length: int, kind: str, seed: int):
    rng = np.random.default_rng(seed)
    U = orthonormal_rows(rng, dim, K)
    w0 = rng.normal(scale=0.5, size=dim).astype(np.float32)
    c, onsets = drift_path(kind, length, rng)
    noise = rng.normal(size=(length, dim)).astype(np.float32)
    x = lfilter([1.0], [1.0, -RHO], noise, axis=0).astype(np.float32) * X_NOISE
    y = x @ w0 + np.einsum("tk,tk->t", x @ U.T, c) + Y_NOISE * rng.normal(size=length).astype(np.float32)
    return x, y.astype(np.float32), c, U, w0, onsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    parser.add_argument("--lengths", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--drifts", nargs="+", default=list(DRIFTS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--full", action="store_true", help="10 seeds everywhere, including D=256 at T=1e5")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    written_mb = 0.0
    for length in args.lengths:
        for dim in args.dims:
            n_seeds = 10 if args.full else seeds_for(length, dim)
            for kind in args.drifts:
                for seed in range(n_seeds):
                    name = f"d{dim}_T{length}_{kind}_s{seed}.npz"
                    path = args.out / name
                    if not path.exists() or args.force:
                        x, y, c, U, w0, onsets = generate_config(dim, length, kind, seed)
                        meta = {
                            "D": dim, "T": length, "drift": kind, "seed": seed, "K": K,
                            "rho": RHO, "x_noise": X_NOISE, "y_noise": Y_NOISE,
                            "n_onsets": int(len(onsets)),
                        }
                        np.savez(path, x=x, y=y, c=c, U=U, w0=w0, onsets=onsets, meta=json.dumps(meta))
                        size_mb = path.stat().st_size / 1e6
                        written_mb += size_mb
                        print(f"[synthetic] {name}: {size_mb:.1f} MB")
                    manifest.append({"file": name, "MB": round(path.stat().st_size / 1e6, 2)})

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[synthetic] {len(manifest)} files, this run +{written_mb:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
