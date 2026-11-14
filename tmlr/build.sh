#!/bin/sh
# Build: pdflatex, bibtex, pdflatex x2. Each step time-limited.
set -e
cd "$(dirname "$0")"
timeout 180 pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || { tail -30 main.log; exit 1; }
timeout 60 bibtex main >/dev/null || { cat main.blg; exit 1; }
timeout 180 pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
timeout 180 pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
# extra pass if LaTeX asks for it
if grep -q "Rerun to get" main.log; then timeout 180 pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null; fi
