#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
cd "$(cd "$(dirname "$0")/.." && pwd)" || exit 1
PY="${BP_PYTHON:-python3}"
"$PY" scripts/bench_bp_tta.py --dataset 2a 2>&1 | tail -20
