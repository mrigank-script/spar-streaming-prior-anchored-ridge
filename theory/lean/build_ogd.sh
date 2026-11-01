#!/bin/bash
# Build the SparTheorems Lean project: copies the committed .lean sources and
# lake configuration into the lake project (default $HOME/SparTheorems;
# override with SPAR_PROJECT=...), resolves Mathlib on first use, and runs
# `lake build`.
set -eu
export PATH="$HOME/.elan/bin:$PATH"
SRC="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${SPAR_PROJECT:-$HOME/SparTheorems}"
mkdir -p "$PROJECT/SparTheorems"
cp "$SRC/SparTheorems/"*.lean "$PROJECT/SparTheorems/"
cp "$SRC/SparTheorems.lean" "$PROJECT/SparTheorems.lean"
cp "$SRC/lakefile.toml" "$SRC/lean-toolchain" "$SRC/lake-manifest.json" "$PROJECT/"
cd "$PROJECT"
if [ ! -d .lake/packages/mathlib ]; then
  lake update
  lake exe cache get
fi
lake build 2>&1 | tail -30