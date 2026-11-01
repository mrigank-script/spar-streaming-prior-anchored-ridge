"""Assemble the release artifact zip for the SPAR paper.

Selects the ship set described in the internal release manifest, drops
process/AI-workflow artifacts and regenerable data, writes a MANIFEST.txt with
sizes and SHA-256 hashes, and zips everything under a single top-level folder.

Console logs, backup CSVs and dead legacy scripts are excluded (logs carry
local absolute paths), and by default the shipped copy of paper/main.tex has its
author block replaced so the supplementary material stays double-blind.  The
manifest hashes the bytes actually written, not the repository originals.

`--named` builds the archival (Zenodo) package instead: the author block is kept
and `--pdf` is shipped as paper/build/main.pdf.  Use it with a separately built
preprint PDF; it never mixes with the anonymous TMLR artifact.

Usage:
    py scripts/package_artifact.py            # -> dist/spar_artifact.zip (anonymous)
    py scripts/package_artifact.py --list     # print the ship set, write nothing
    py scripts/package_artifact.py --out other.zip
    py scripts/package_artifact.py --named --pdf dist/spar_named.pdf
                                              # -> dist/spar_zenodo.zip (named)
"""

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files shipped verbatim from the repository root.
ROOT_FILES = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-wsl.txt",
]

# Whole trees that ship (Python sources only, caches stripped).
SOURCE_DIRS = ["core", "dataio", "runner", "scripts", "tests"]

# paper/: sources + style + the compiled PDF; no aux scratch.
PAPER_FILES = [
    "main.tex",
    "appendix.tex",
    "references.bib",
    "tmlr.sty",
    "tmlr.bst",
    "build.cmd",
    "build.sh",
]

# Lean formalization: sources and build scripts.
LEAN_DIR = Path("theory") / "lean"
LEAN_FILES = ["lakefile.toml", "lean-toolchain", "lake-manifest.json",
              "SparTheorems.lean", "build_ogd.sh", "build_err.sh",
              "print_axioms.sh", "axioms.log"]
LEAN_SRC_DIR = LEAN_DIR / "SparTheorems"

# Substrings anywhere in a relative path that force exclusion.
EXCLUDE_PARTS = (
    "__pycache__",
    ".pytest_cache",
    "_audit",
    "redteam",
    "ch3_theory.tex",
    "ch4_spar_theorems.tex",
    "package_artifact.py",
    ".backup",            # eeg_tier1_*.backup.csv
)
# Exact relative paths excluded from an otherwise-shipped tree.
EXCLUDE_FILES = {
    "results/redteam",
    # Legacy scripts (and their outputs) that the paper, README, run_all.py and
    # tests never reference; `bench_quant`/`bench_finance2` also need finance
    # data that is not shipped, so they cannot run from the artifact.
    "scripts/bench_quant.py",
    "scripts/bench_finance2.py",
    "scripts/bench_runner.py",
    "scripts/sweep_synthetic.py",
    "results/quant_results.csv",
    "results/synthetic_sweep.csv",
    "results/factorial_mismatch_legacy.csv",
}
# File suffixes never shipped even if present.  Console logs are dropped: they
# carry local absolute paths and machine usernames.
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log")


def _skip(rel: Path) -> bool:
    s = rel.as_posix()
    if rel.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if s in EXCLUDE_FILES:
        return True
    if DROP_EEG and s.startswith("data/eeg_cache/"):
        return True
    return any(part in s for part in EXCLUDE_PARTS)


def _tree(base: Path, rel_root: Path):
    if not base.exists():
        return
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = rel_root / p.relative_to(base)
        if not _skip(rel):
            yield rel, p


def ship_set():
    """Yield (relative_path, absolute_path) for everything that ships."""
    for name in ROOT_FILES:
        p = ROOT / name
        if p.exists():
            yield Path(name), p

    for d in SOURCE_DIRS:
        yield from _tree(ROOT / d, Path(d))

    for name in PAPER_FILES:
        p = ROOT / "paper" / name
        if p.exists():
            yield Path("paper") / name, p
    pdf = ROOT / "paper" / "build" / "main.pdf"
    if pdf.exists():
        yield Path("paper") / "build" / "main.pdf", pdf

    for name in LEAN_FILES:
        p = LEAN_DIR / name
        if p.exists():
            yield LEAN_DIR / name, p
    yield from _tree(LEAN_SRC_DIR, LEAN_SRC_DIR)

    yield from _tree(ROOT / "results", Path("results"))
    yield from _tree(ROOT / "data" / "eeg_cache", Path("data") / "eeg_cache")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The repository keeps the real author block (needed for the named camera-ready
# build with \usepackage[preprint]{tmlr}).  The artifact is double-blind
# supplementary material, so the shipped copy of main.tex is anonymized.
ANON_AUTHOR = "\\author{Anonymous submission}"
_AUTHOR_BLOCK = re.compile(r"\\author\{.*?\n\}", re.S)
_PII_TOKENS = ("Mrigank", "DLDAV", "Shalimar", "mrigank.codes")

# Set by main() from --named / --pdf.
NAMED = False
NAMED_PDF = None
# Set by main() from --no-eeg-cache (TMLR's 100 MB supplementary cap).
DROP_EEG = False


def shipped_bytes(rel: Path, path: Path):
    """Replacement bytes for a shipped file, or None to ship it verbatim."""
    s = rel.as_posix()
    if s == "paper/build/main.pdf" and NAMED:
        if NAMED_PDF is None or not NAMED_PDF.exists():
            raise SystemExit(f"package_artifact: --named needs --pdf ({NAMED_PDF})")
        return NAMED_PDF.read_bytes()
    if s == "README.md" and NAMED:
        text = path.read_text(encoding="utf-8")
        return text.replace(
            "the submission PDF and the packaged source are anonymous.",
            "this archival package ships the named PDF and source.",
        ).encode("utf-8")
    if s == "paper/main.tex" and NAMED:
        text = path.read_text(encoding="utf-8")
        # Replace only the active command; the comment above it already mentions
        # \usepackage[preprint]{tmlr}, so a substring test would be fooled.
        if "\\usepackage{tmlr}" in text:
            text = text.replace("\\usepackage{tmlr}",
                                "\\usepackage[preprint]{tmlr}", 1)
        active = [ln.strip() for ln in text.splitlines()]
        if "\\usepackage[preprint]{tmlr}" not in active:
            raise SystemExit("package_artifact: could not enable the preprint option")
        return text.encode("utf-8")
    if s != "paper/main.tex" or NAMED:
        return None
    # A callable replacement avoids re's backslash-escape processing (\a would
    # otherwise become a bell character and corrupt \author).
    text, n = _AUTHOR_BLOCK.subn(lambda _m: ANON_AUTHOR,
                                 path.read_text(encoding="utf-8"), count=1)
    if n != 1 or ANON_AUTHOR not in text:
        raise SystemExit("package_artifact: author block not anonymized in main.tex")
    for token in _PII_TOKENS:
        if token.lower() in text.lower():
            raise SystemExit(f"package_artifact: {token!r} survives anonymization")
    return text.encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output zip (default: dist/spar_artifact.zip, or "
                         "dist/spar_zenodo.zip with --named)")
    ap.add_argument("--named", action="store_true",
                    help="archival (Zenodo) build: keep the author block and "
                         "ship the named PDF instead of the anonymous one")
    ap.add_argument("--pdf", help="named PDF to ship as paper/build/main.pdf")
    ap.add_argument("--no-eeg-cache", action="store_true",
                    help="drop data/eeg_cache (TMLR's 100 MB supplementary cap)")
    ap.add_argument("--list", action="store_true", help="print the ship set only")
    args = ap.parse_args()

    global NAMED, NAMED_PDF, DROP_EEG
    NAMED = args.named
    DROP_EEG = args.no_eeg_cache
    if NAMED:
        NAMED_PDF = Path(args.pdf) if args.pdf else ROOT / "dist" / "spar_named.pdf"
    if args.out:
        out = Path(args.out)
    elif NAMED:
        out = ROOT / "dist" / "spar_zenodo.zip"
    elif DROP_EEG:
        out = ROOT / "dist" / "spar_tmlr_supplementary.zip"
    else:
        out = ROOT / "dist" / "spar_artifact.zip"

    items = list(ship_set())
    if not items:
        print("no files selected", file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for _, p in items)
    print(f"ship set: {len(items)} files, {total / 1e6:.1f} MB uncompressed")

    if args.list:
        for rel, _ in items:
            print(f"  {rel.as_posix()}")
        return 0

    top = "spar" if NAMED else "spar_artifact"
    manifest = ["SPAR paper artifact — file list, sizes, SHA-256", ""]
    for rel, p in items:
        data = shipped_bytes(rel, p)
        if data is None:
            manifest.append(f"{sha256(p)}  {p.stat().st_size:>12d}  {rel.as_posix()}")
        else:
            manifest.append(
                f"{hashlib.sha256(data).hexdigest()}  {len(data):>12d}  {rel.as_posix()}")
    manifest_text = "\n".join(manifest) + "\n"

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, p in items:
            arc = f"{top}/{rel.as_posix()}"
            data = shipped_bytes(rel, p)
            if data is None:
                zf.write(p, arc)
            else:
                print(f"overrode {rel.as_posix()} for the shipped copy")
                zf.writestr(arc, data)
        zf.writestr(f"{top}/MANIFEST.txt", manifest_text)

    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
