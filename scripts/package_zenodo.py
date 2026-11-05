"""Assemble the Zenodo archival package for the JMLR manuscript.

Ships the Python codebase, the Lean formalization, the machine-printed results,
the JMLR LaTeX sources, and the de-branded preprint PDFs, with a MANIFEST of
sizes and SHA-256 hashes.  The preprint PDFs are the same sources built with
``\\usepackage[preprint]{jmlr2e}`` (journal line and editor field dropped, same
layout); ``paper/jmlr/build.cmd`` is the plain submission build.

The ship set is explicit.  One-off/legacy scripts and their orphan outputs that
no pipeline stage, README, paper citation, or import reaches are excluded, as
are binary intermediates (``.npy``) and console logs.

Usage:
    py scripts/package_zenodo.py \\
        --preprint dist/spar_preprint.pdf \\
        --preprint-online dist/spar_preprint_online_appendix.pdf
"""

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JMLR_DIR = Path("paper") / "jmlr"
LEAN_DIR = Path("theory") / "lean"
LEAN_SRC = LEAN_DIR / "SparTheorems"

ROOT_FILES = ["README.md", "CHANGELOG.md", "VERSION", "pyproject.toml",
              "requirements.txt", "requirements-wsl.txt"]
SOURCE_DIRS = ["core", "dataio", "runner", "scripts", "tests"]
LEAN_FILES = ["lakefile.toml", "lean-toolchain", "lake-manifest.json",
              "SparTheorems.lean", "build_ogd.sh", "build_err.sh",
              "print_axioms.sh", "axioms.log"]
JMLR_PAPER_FILES = ["main.tex", "appendix.tex", "appendix_online.tex",
                    "references.bib", "jmlr2e.sty", "build.cmd", "build.sh"]

# Legacy/one-off files that reach no pipeline stage, README, paper citation, or
# import, plus binary intermediates and logs.
EXCLUDE_PATHS = {
    # scripts/ : orphaned experiments
    "scripts/bench_align2.py",
    "scripts/metrics_suite.py",
    "scripts/pilot_units.py",
    "scripts/package_artifact.py",   # TMLR-only packager, superseded
    # results/ : orphan outputs and stray binaries
    "results/eeg_tier1_2a.backup.csv",
    "results/eeg_tier1_2b.backup.csv",
}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log", ".npy", ".bak")
EXCLUDE_PARTS = ("__pycache__", ".pytest_cache", "_audit", "redteam",
                 ".backup")


def _skip(rel: Path) -> bool:
    s = rel.as_posix()
    if rel.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if s in EXCLUDE_PATHS:
        return True
    return any(part in s for part in EXCLUDE_PARTS)


def _tree(base: Path, rel_root: Path):
    if not base.exists():
        return
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = rel_root / p.relative_to(base)
            if not _skip(rel):
                yield rel, p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ship_set(preprint: Path, preprint_online: Path):
    """Yield (relative_path, absolute_path) for the Zenodo package."""
    for name in ROOT_FILES:
        p = ROOT / name
        if p.exists():
            yield Path(name), p
    for d in SOURCE_DIRS:
        yield from _tree(ROOT / d, Path(d))
    for name in JMLR_PAPER_FILES:
        p = ROOT / JMLR_DIR / name
        if p.exists():
            yield JMLR_DIR / name, p
    yield JMLR_DIR / "build" / "main.pdf", preprint
    yield JMLR_DIR / "build" / "appendix_online.pdf", preprint_online
    for name in LEAN_FILES:
        p = ROOT / LEAN_DIR / name
        if p.exists():
            yield LEAN_DIR / name, p
    yield from _tree(ROOT / LEAN_SRC, LEAN_SRC)
    yield from _tree(ROOT / "results", Path("results"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprint", required=True,
                    help="de-branded manuscript PDF")
    ap.add_argument("--preprint-online", required=True,
                    help="de-branded online-appendix PDF")
    ap.add_argument("--out", default=str(ROOT / "dist" / "spar_zenodo.zip"))
    args = ap.parse_args()

    preprint = Path(args.preprint)
    preprint_online = Path(args.preprint_online)
    for p in (preprint, preprint_online):
        if not p.exists():
            sys.exit(f"package_zenodo: missing PDF {p}")

    items = list(ship_set(preprint, preprint_online))
    total = sum(p.stat().st_size for _, p in items)
    print(f"ship set: {len(items)} files, {total / 1e6:.1f} MB uncompressed")

    manifest = ["SPAR paper artifact (Zenodo) - file list, sizes, SHA-256", ""]
    for rel, p in items:
        manifest.append(f"{sha256(p)}  {p.stat().st_size:>12d}  {rel.as_posix()}")
    manifest_text = "\n".join(manifest) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, p in items:
            zf.write(p, f"spar/{rel.as_posix()}")
        zf.writestr("spar/MANIFEST.txt", manifest_text)

    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
