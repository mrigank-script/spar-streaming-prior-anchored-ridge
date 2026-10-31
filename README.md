# Streaming Prior-Anchored Ridge (SPAR)

Closed-form streaming heads for drifting data streams: a ridge solve anchored
to a prior, with two quantities the method calibrates online from its own
residuals — the anchor strength and the forgetting window. The repository also
contains the formalization of the proof core in Lean 4.

This is the companion code for the manuscript in `paper/` (`paper/main.tex`,
built with `paper/build.cmd`). Every number and figure in the paper is
machine-printed into `results/` by the scripts below.

## Layout

| Path | Contents |
|---|---|
| `core/` | math primitives: metrics, features (stats/lag/FFT/LRU), LRU scan, ridge, alignment |
| `dataio/` | data loaders: synthetic drift suite, finance panels, BCI IV (2a/2b), time-series panels |
| `runner/` | streaming arms (RLS, re-anchoring, spectral anchor), classifier head, live trigger loop |
| `scripts/` | benchmarks, experiments, and the reproducibility driver |
| `results/` | every table and figure in the paper, machine-printed from the drivers |
| `theory/lean/` | the Lean 4 formalization (`SparTheorems`) and its build scripts |
| `paper/` | the manuscript source (`main.tex`, `appendix.tex`) and the TMLR style files |
| `tests/` | pytest suite |

The umbrella package is named `aal` in `pyproject.toml`; SPAR is the method it
contributes. Internal Python/CSV identifiers are frozen for provenance:

| Display name | Meaning | Internal IDs |
|---|---|---|
| SPAR-Pull | main method: anchor strength + forgetting window, variance-floor-subtracted drift inversion | `spar_pull`, `spar_kf2` |
| SPAR-Pull-S | scalar-strength arm | `spar_scalar` |
| SPAR-Pull-D | per-direction-strength eigenbasis head | `spar_diag` |
| SPAR-EG | Hedge composition over leaky-window experts | `spar_eg` |
| SPAR-EG+ | SPAR-EG pool augmented with the SPAR-Pull trajectory | `spar_eg_plus` |
| SPAR-Pull-legacy | earlier parameter-unit strength variant, retained in the CSVs | `spar_kf` |

## 1. Environment

Python 3.12 with the pinned dependencies:

```bash
py -m pip install -r requirements.txt      # Windows launcher
# or: python3 -m pip install -r requirements.txt
```

Float64 is required for the streaming solves and is enabled by the drivers via
`JAX_ENABLE_X64=1`; a CPU runtime is recommended for the fp64 streams. The
backprop-only baselines additionally need a GPU environment — see §6.

This README writes `py` for the Windows launcher; on Linux/macOS/WSL every
command is identical with `python3` in its place.

## 2. Get the data

The synthetic results need no data; the real-data sections each need the block
below, fetched once.

### Synthetic drift suite (no download)

```bash
py scripts/generate_synthetic.py      # writes data/synthetic/*.npz from pinned seeds
```

### EEG — BCI IV 2a and 2b

```bash
py scripts/download_eeg.py            # moabb: BNCI2014_001 (2a) + BNCI2014_004 (2b)
py scripts/download_eeg_manual.py     # resumable fallback for the slow 2b server
py scripts/export_eeg_npz.py          # writes data/eeg_cache/{2a,2b}/subject*.npz
```

Raw files land in the MNE data directory (`MNE_DATA` env var or the mne config,
default `~/mne_data`) under `MNE-bnci-data/~bci/database/`. The EEG results in
`results/` were produced from a pre-built `data/eeg_cache/`; run the commands
above once to regenerate it, after which the EEG results rerun without the
download.

### Time-series panels — Electricity, Traffic, Weather, ETT, Exchange

Download the well-preprocessed TSLib/Autoformer files once and drop them into
`data/ts/`:

- Google Drive: <https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2>
- Hugging Face: <https://huggingface.co/datasets/thuml/Time-Series-Library>

Place them with the names the loaders expect:

```
data/ts/electricity.csv   data/ts/traffic.csv   data/ts/weather.csv
data/ts/exchange.txt      data/ts/etth1.csv     data/ts/ettm1.csv   (and etth2/ettm2)
```

The paper's real-stream result (`scripts/bench_stream_real.py`) reads
`data/ts/electricity.csv` — the hourly-load panel, 321 clients × 26,304 rows.
The loader accepts either `electricity.csv` or `electricity.txt`.

### Finance (optional, legacy benchmarks)

```bash
py scripts/download_finance.py        # Ken French daily factors + ETF panel
py scripts/download_panel.py          # daily S&P 500 equity panel (yfinance)
```

## 3. Reproduce the paper

One driver runs every stage, and each stage is resumable.

```bash
py scripts/run_all.py --list           # list the pipeline steps
py scripts/run_all.py --steps quick    # verify + aggregate + figures (~2 min)
py scripts/run_all.py                  # the full grid (hours; resumable)
```

Stages: `verify` (exactness checks + sympy identities; exits nonzero on any
failed identity), `factorial` (the grid), `sparkf`/`append` (the appended grid
arms), `aggregate`, `ablation`, `falseesc`, `regret`, `ops` (per-step cost — run
alone for clean timing), `eeg`, `bp` (GPU only, §6), `figures`.

The revision grid (prediction-unit SPAR, expanded baseline sweeps) and its
ablations:

```bash
py scripts/revision_runs.py --suite grid
py scripts/revision_runs.py --suite ablation
py scripts/summarize_revision.py
```

The real electricity stream (three held-out clients), the transient sweep, and
all numerical diagnostics:

```bash
py scripts/bench_stream_real.py --clients 11,82,171   # -> results/stream_electricity.csv
py scripts/transient_test.py                          # -> results/transient_test.csv
py scripts/diagnostics.py --diag all                  # vff|windup|sgd|coupling|rmt|alt_trust
```

## 4. Build the manuscript

From `paper/`:

```bash
build.cmd          # Windows: 4-pass pdflatex + bibtex -> build/main.pdf
bash build.sh      # Linux / macOS / WSL: same build -> build/main.pdf
```

Both scripts create `build/` if it is missing and clear stray root-level
outputs first, so the result is always `paper/build/main.pdf` (never a root
`main.pdf`). The compiled PDF also ships with the artifact. `main.tex` keeps
the author block for the named camera-ready (`\usepackage[preprint]{tmlr}`);
this archival package ships the named PDF and source.

## 5. Lean artifact (optional)

The proofs in `theory/lean/SparTheorems/` build with Lean 4 + Mathlib via
elan/lake, in WSL or any Linux/macOS box:

```bash
bash theory/lean/build_ogd.sh      # copies the sources into $HOME/SparTheorems and builds
bash theory/lean/print_axioms.sh   # prints the axioms each headline theorem rests on
```

Set `SPAR_PROJECT` to build into a different lake project directory. First-time
setup (`lakefile.toml`, `lake exe cache get`) is given in the appendix recipe.

## 6. GPU backprop baselines (optional)

The BP reference arms (EEGNet-lite, causal WiSE-FT) run on a WSL GPU venv:

```bash
wsl -e bash scripts/run_bp_tta_2a.sh
wsl -e bash scripts/run_bp_tta_2b.sh
```

Set `BP_PYTHON` to the venv interpreter (default `python3`); the GPU extras are
in `requirements-wsl.txt`. On Windows, do not call `moabb` downloads directly
(path-sanitizer bug); the loader self-heals.

## 7. Tests

```bash
py -m pytest tests -q        # Windows
python3 -m pytest tests -q   # Linux / macOS
```

All tests pass; the EEG-specific cases skip automatically when `mne` or
`pyriemann` is not installed.
