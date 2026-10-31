#!/bin/bash
# Build the SparTheorems Lean project: copies the committed .lean sources into
# the lake project (default $HOME/SparTheorems; override with SPAR_PROJECT=...),
# generates the root import file, and runs lake build. First-time setup
# (lakefile.toml, Mathlib cache) is described in the paper's appendix recipe.
set -eu
export PATH="$HOME/.elan/bin:$PATH"
SRC="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${SPAR_PROJECT:-$HOME/SparTheorems}"
mkdir -p "$PROJECT/SparTheorems"
cp "$SRC/SparTheorems/"*.lean "$PROJECT/SparTheorems/"
printf 'import SparTheorems.Core\nimport SparTheorems.Ogd\nimport SparTheorems.Hedge\nimport SparTheorems.BoxD\nimport SparTheorems.Coupled\nimport SparTheorems.TestAx\n' > "$PROJECT/SparTheorems.lean"
cd "$PROJECT"
lake build 2>&1 | tail -30
