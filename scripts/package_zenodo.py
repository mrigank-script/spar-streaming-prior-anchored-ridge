"""Assemble the Zenodo archival package for the JMLR manuscript.

Ships the Python codebase, the Lean formalization, the machine-printed results,
the JMLR LaTeX sources, and the de-branded preprint PDFs, with a MANIFEST of
sizes and SHA-256 hashes.  The preprint PDFs are the same sources built with
``\\usepackage[preprint]{jmlr2e}`` (journal line and editor field dropped, same
layout); ``paper/jmlr/build.cmd`` is the plain submission build.

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

import package_artifact as pa

ROOT = pa.ROOT
JMLR_DIR = Path("paper") / "jmlr"
JMLR_PAPER_FILES = [
    "main.tex", "appendix.tex", "appendix_online.tex", "references.bib",
    "jmlr2e.sty", "build.cmd", "build.sh",
    "cover_letter.txt", "cover_letter.tex",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ship_set(preprint: Path, preprint_online: Path):
    """Yield (relative_path, absolute_path) for the Zenodo package."""
    for name in pa.ROOT_FILES:
        p = ROOT / name
        if p.exists():
            yield Path(name), p
    for d in pa.SOURCE_DIRS:
        yield from pa._tree(ROOT / d, Path(d))
    for name in JMLR_PAPER_FILES:
        p = ROOT / JMLR_DIR / name
        if p.exists():
            yield JMLR_DIR / name, p
    yield JMLR_DIR / "build" / "main.pdf", preprint
    yield JMLR_DIR / "build" / "appendix_online.pdf", preprint_online
    for name in pa.LEAN_FILES:
        p = ROOT / pa.LEAN_DIR / name
        if p.exists():
            yield pa.LEAN_DIR / name, p
    yield from pa._tree(ROOT / pa.LEAN_SRC_DIR, pa.LEAN_SRC_DIR)
    yield from pa._tree(ROOT / "results", Path("results"))


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
