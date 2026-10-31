@echo off
rem Builds the manuscript with fully resolved references (4 passes).
rem Output: build\main.pdf  (never a root main.pdf)
rem Stray root outputs shadow the build directory and break citations.
del /q main.aux main.log main.out main.pdf main.synctex.gz 2>nul
if not exist build mkdir build
pdflatex -interaction=nonstopmode -output-directory=build main.tex
bibtex build\main
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
echo Done: build\main.pdf
