"""Build the de-branded preprint PDFs from the JMLR sources.

Copies ``paper/jmlr/`` into a scratch tree, switches ``\\usepackage{jmlr2e}`` to
``\\usepackage[preprint]{jmlr2e}`` (jmlr2e's official option: it empties the
journal header and drops the editor field, leaving the layout unchanged), runs
the same multi-pass pdflatex+bibtex order as ``paper/jmlr/build.cmd``, and
writes the two PDFs to ``dist/``.  Nothing in the repository sources changes.

Usage:
    py scripts/build_preprint.py
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "jmlr"
DIST = ROOT / "dist"

COPIES = ["main.tex", "appendix.tex", "appendix_online.tex", "references.bib",
          "jmlr2e.sty"]
DEBRAND = ["main.tex", "appendix_online.tex"]


def compile_worktree(work: Path) -> None:
    paper = work / "paper"
    paper.mkdir(parents=True)
    (paper / "build").mkdir()
    (paper / "figures").mkdir()
    for name in COPIES:
        shutil.copy2(SRC / name, paper / name)
    for fig in (ROOT / "results" / "figures").glob("*.pdf"):
        shutil.copy2(fig, paper / "figures" / fig.name)

    for name in DEBRAND:
        p = paper / name
        text = p.read_text(encoding="utf-8")
        if "\\usepackage{jmlr2e}" not in text:
            sys.exit(f"build_preprint: {name} does not load jmlr2e")
        text = text.replace("\\usepackage{jmlr2e}",
                            "\\usepackage[preprint]{jmlr2e}", 1)
        p.write_text(text, encoding="utf-8", newline="")

    tex = ["pdflatex", "-interaction=nonstopmode",
           "-output-directory=build", "main.tex"]
    tex_onl = ["pdflatex", "-interaction=nonstopmode",
               "-output-directory=build", "appendix_online.tex"]
    for cmd in (tex, tex_onl,
                ["bibtex", "build/main"], ["bibtex", "build/appendix_online"],
                tex, tex, tex_onl, tex_onl):
        subprocess.run(cmd, cwd=paper, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    for tag, log in (("main", "main.log"), ("online", "appendix_online.log")):
        text = (paper / "build" / log).read_text(encoding="utf-8",
                                                 errors="replace")
        errs = sum(1 for ln in text.splitlines() if ln.startswith("!"))
        undef = len(re.findall(r"Reference .* undefined", text))
        print(f"  {tag}: errors={errs} undefined-refs={undef}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        compile_worktree(Path(tmp))
        paper = Path(tmp) / "paper" / "build"
        DIST.mkdir(exist_ok=True)
        targets = {
            "main.pdf": DIST / "spar_preprint.pdf",
            "appendix_online.pdf": DIST / "spar_preprint_online_appendix.pdf",
        }
        for src_name, dst in targets.items():
            shutil.copy2(paper / src_name, dst)
            print(f"wrote {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
