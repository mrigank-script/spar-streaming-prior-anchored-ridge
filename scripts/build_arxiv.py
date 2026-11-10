"""Build the combined arXiv preprint from the JMLR sources.

Merges the de-branded main preprint (``paper/jmlr/main.tex`` with the
``preprint`` style option) and the online appendix into a single document, and
writes a self-contained source folder to ``dist/arxiv/``: the combined
``main.tex``, ``appendix.tex``, the online-appendix body ``appendix_online.tex``,
``references.bib`` + ``main.bbl``, the ``jmlr2e.sty`` style file, and the figure
PDFs the three documents use.  No code is shipped.

It then compiles the folder and writes ``dist/spar_arxiv.pdf`` and
``dist/spar_arxiv_source.zip``.

Usage:
    py scripts/build_arxiv.py
"""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "jmlr"
OUT = ROOT / "dist" / "arxiv"
DIST = ROOT / "dist"

FIGURES = ["F1_schematic", "F2_geometry", "F3_lambda_cliff", "F4_two_dials",
           "F5_noise_floor", "F6_prior_immunity", "F7_regret_scaling"]

RESTATED = None  # appendix.tex already defines the restated environment


def build_sources() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)

    shutil.copy2(SRC / "jmlr2e.sty", OUT / "jmlr2e.sty")
    shutil.copy2(SRC / "references.bib", OUT / "references.bib")
    shutil.copy2(SRC / "appendix.tex", OUT / "appendix.tex")
    for name in FIGURES:
        shutil.copy2(ROOT / "results" / "figures" / f"{name}.pdf",
                     OUT / "figures" / f"{name}.pdf")

    # --- combined main document ---------------------------------------------
    main = (SRC / "main.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
    if "\\usepackage{jmlr2e}" not in main:
        raise SystemExit("main.tex does not load jmlr2e")
    main = main.replace("\\usepackage{jmlr2e}",
                        "\\usepackage[preprint]{jmlr2e}", 1)
    # All labels live in this one document now; the xr external references go.
    main = main.replace("\\usepackage{xr}\n\\externaldocument{appendix_online}\n", "")
    # Pull the online appendix in ahead of the single bibliography.
    main = main.replace("\\input{appendix}\n\n\\bibliography{references}",
                        "\\input{appendix}\n\\input{appendix_online}\n\n"
                        "\\bibliography{references}")
    for needle in ("\\usepackage[preprint]{jmlr2e}", "\\input{appendix_online}"):
        if needle not in main:
            raise SystemExit(f"merge failed: {needle!r} missing")
    (OUT / "main.tex").write_text(main, encoding="utf-8", newline="")

    # --- online appendix body (preamble and bibliography stripped) ----------
    online = (SRC / "appendix_online.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
    start = online.index("\\maketitle") + len("\\maketitle")
    end = online.index("\\bibliography{references}")
    body = online[start:end].strip() + "\n"
    if "\\documentclass" in body or "\\begin{document}" in body:
        raise SystemExit("online appendix body still carries a preamble")
    (OUT / "appendix_online.tex").write_text(body, encoding="utf-8",
                                             newline="")


def compile_pdf() -> None:
    tex = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
    for cmd in (tex, ["bibtex", "main"], tex, tex):
        subprocess.run(cmd, cwd=OUT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    log = (OUT / "main.log").read_text(encoding="utf-8", errors="replace")
    errs = [ln for ln in log.splitlines() if ln.startswith("!")]
    undef = len(re.findall(r"Reference .*? undefined", log))
    dupes = len(re.findall(r"Label .*? multiply defined", log))
    print(f"  errors={len(errs)} undefined-refs={undef} multiply-defined={dupes}")
    for e in errs[:10]:
        print("   ", e[:120])


def main() -> int:
    build_sources()
    compile_pdf()
    pdf = OUT / "main.pdf"
    shutil.copy2(pdf, DIST / "spar_arxiv.pdf")
    print("wrote", (DIST / "spar_arxiv.pdf").relative_to(ROOT))

    zip_path = DIST / "spar_arxiv_source.zip"
    wanted = ["main.tex", "appendix.tex", "appendix_online.tex",
              "references.bib", "main.bbl", "jmlr2e.sty"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in wanted:
            f = OUT / name
            if f.exists():
                z.write(f, name)
        for fig in sorted((OUT / "figures").glob("*.pdf")):
            z.write(fig, "figures/" + fig.name)
    print("wrote", zip_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
