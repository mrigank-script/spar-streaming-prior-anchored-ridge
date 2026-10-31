"""Export epoched BCI IV trials to npz so other environments (e.g., WSL GPU) can use them."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataio import load_eeg_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2a", choices=["2a", "2b"])
    parser.add_argument("--subjects", type=int, default=9)
    args = parser.parse_args()
    out_dir = ROOT / "data" / "eeg_cache" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for subject in range(1, args.subjects + 1):
        out = load_eeg_subject(dataset=args.dataset, subject=subject)
        (X_train, y_train), (X_stream, y_stream) = out["train"], out["stream"]
        np.savez_compressed(
            out_dir / f"subject{subject}.npz",
            X_train=X_train, y_train=y_train, X_stream=X_stream, y_stream=y_stream,
        )
        print(f"exported {args.dataset} subject {subject}: train={X_train.shape} stream={X_stream.shape}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
