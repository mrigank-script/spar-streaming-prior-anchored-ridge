#!/bin/bash
# Build the SparTheorems Lean project and report any errors verbosely.
# Same conventions as build_ogd.sh (SPAR_PROJECT overrides the project dir).
set -eu
export PATH="$HOME/.elan/bin:$PATH"
SRC="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${SPAR_PROJECT:-$HOME/SparTheorems}"
mkdir -p "$PROJECT/SparTheorems"
cp "$SRC/SparTheorems/"*.lean "$PROJECT/SparTheorems/"
printf 'import SparTheorems.Core\nimport SparTheorems.Ogd\nimport SparTheorems.Hedge\nimport SparTheorems.BoxD\nimport SparTheorems.Coupled\nimport SparTheorems.TestAx\n' > "$PROJECT/SparTheorems.lean"
cd "$PROJECT"
lake build 2>&1 | grep -B 2 -A 8 "^error:" | tail -40
echo "=== BUILD RESULT ==="
lake build 2>&1 | tail -2
