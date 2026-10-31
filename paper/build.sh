#!/usr/bin/env bash
# Builds the manuscript with fully resolved references (4 passes).
# Output: build/main.pdf  (never a root main.pdf).
# Stray root outputs shadow the build directory and break citations.
# POSIX twin of build.cmd (Linux/macOS/WSL).
set -eu
cd "$(dirname "$0")"
rm -f main.aux main.log main.out main.pdf main.synctex.gz
mkdir -p build
pdflatex -interaction=nonstopmode -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
echo "Done: build/main.pdf"
