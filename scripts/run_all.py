"""One-command reproducibility driver: regenerates every table and figure.

Usage:
  py scripts/run_all.py                 # everything (hours; resumable)
  py scripts/run_all.py --steps quick    # sanity + aggregation + figures
  py scripts/run_all.py --steps verify,factorial,aggregate
  py scripts/run_all.py --list

Steps:
  verify      baseline library sanity (exactness checks) + sympy identities
  factorial   31-config grid chunks (resumable; skips done combos)
  sparkf      append the SPAR-Pull grid arm to the grid
  append      all appended-grid arms (SPAR-Pull grid arm, SPAR-EG, estimated-R
              Kalman, VFF both forms, both OGD arms, SPAR-EG+, causal-q
              Kalman arms, mismatch suite, VFF diagnostics, damped trace)
  aggregate   family-best aggregation + robustness analysis
  ablation    SPAR-Pull component ablation
  falseesc    Proposition 3 false-escalation calibration
  regret      Proposition 4 regret scaling (horizon sweep + dimension sweep)
  ops         per-step cost table (run ALONE for clean timing)
  eeg         BCI IV 2a + 2b Tier-1 benchmark (Windows, cached data)
  bp          BP arms (requires WSL GPU: run run_bp_tta_2a.sh / _2b.sh)
  figures     all paper figures
"""

import argparse
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "4")

STEPS = ["verify", "factorial", "sparkf", "append", "aggregate", "ablation",
         "falseesc", "regret", "ops", "eeg", "bp", "figures"]
QUICK_STEPS = ["verify", "aggregate", "figures"]

APPEND_SCRIPTS = [
    "append_arms.py",
    "regen_spar_eg_plus.py",
    "revision_runs.py --suite grid",
    "append_damped_trace.py",
    "mismatch_suite.py",
    "diagnostics.py",
]


def run(cmd, label, env=None):
    print(f"\n=== [{label}] {' '.join(cmd)} ===", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    dt = time.perf_counter() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"=== [{label}] {status} in {dt:.0f}s ===", flush=True)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="all",
                        help="comma list, or 'all', or 'quick'")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    steps = STEPS
    if args.list:
        print("\n".join(steps))
        return
    if args.steps == "quick":
        chosen = QUICK_STEPS
    elif args.steps == "all":
        chosen = steps
    else:
        chosen = args.steps.split(",")

    py = sys.executable
    for name in chosen:
        if name == "verify":
            codes = (run([py, "scripts/rls_baselines.py"], "verify:baselines"),
                     run([py, "scripts/verify_spar_theorems.py"], "verify:sympy"))
            if any(code != 0 for code in codes):
                sys.exit(1)
        elif name == "factorial":
            for drift in ("abrupt", "covrot", "both"):
                run([py, "scripts/synthetic_factorial.py", "--suite", "main",
                     "--drifts", drift, "--tag", f"d_{drift}"],
                    f"factorial:{drift}")
            run([py, "scripts/synthetic_factorial.py", "--suite", "extra",
                 "--tag", "extras"], "factorial:extras")
        elif name == "sparkf":
            run([py, "scripts/append_spar_kf.py"], "spar-pull-grid")
        elif name == "append":
            for script in APPEND_SCRIPTS:
                parts = script.split()
                run([py, f"scripts/{parts[0]}", *parts[1:]],
                    f"append:{parts[0]}")
        elif name == "aggregate":
            run([py, "scripts/aggregate_factorial.py"], "aggregate")
        elif name == "ablation":
            run([py, "scripts/revision_runs.py", "--suite", "ablation"],
                "ablation")
        elif name == "falseesc":
            run([py, "scripts/false_esc_test.py"], "false-escalation")
        elif name == "regret":
            run([py, "scripts/regret_scaling.py"], "regret-scaling")
            run([py, "scripts/regret_dscaling.py"], "regret-dimension-sweep")
        elif name == "ops":
            run([py, "scripts/bench_ops.py"], "ops-table")
        elif name == "eeg":
            for ds in ("2a", "2b"):
                run([py, "scripts/bench_eeg.py", "--dataset", ds],
                    f"eeg:baseline-{ds}")
                run([py, "scripts/bench_eeg_tier1.py", "--dataset", ds],
                    f"eeg:{ds}")
            run([py, "scripts/bench_spar_eeg.py", "--dataset", "2a"],
                "eeg:spar-legacy-2a")
            run([py, "scripts/bench_spar_eeg.py", "--dataset", "2b",
                 "--subjects", "9"], "eeg:spar-legacy-2b")
        elif name == "bp":
            print("BP arms run on the WSL GPU venv (from the repository root):")
            print("  wsl -e bash scripts/run_bp_tta_2a.sh")
            print("  wsl -e bash scripts/run_bp_tta_2b.sh")
        elif name == "figures":
            run([py, "scripts/make_figures.py"], "figures")
        else:
            print(f"unknown step: {name}")


if __name__ == "__main__":
    main()
