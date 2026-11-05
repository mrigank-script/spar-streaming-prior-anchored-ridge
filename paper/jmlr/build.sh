#!/bin/bash
# Builds the JMLR manuscript and its online appendix (cross-referenced).
# Outputs: build/main.pdf  and  build/appendix_online.pdf
set -eu
mkdir -p build
rm -f build/main.aux build/main.bbl build/main.blg build/main.out build/main.pdf
rm -f build/appendix_online.aux build/appendix_online.bbl build/appendix_online.blg build/appendix_online.out build/appendix_online.pdf
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build appendix_online.tex
bibtex build/main
bibtex build/appendix_online
# Three interleaved passes so the xr cross-references and the LastPage
# header (the JMLR page range) settle between the two documents.
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build appendix_online.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build appendix_online.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build appendix_online.tex
echo "Done: build/main.pdf and build/appendix_online.pdf"
