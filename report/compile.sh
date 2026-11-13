#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Compiling methodology report..."

if command -v /opt/homebrew/bin/tectonic >/dev/null 2>&1; then
    /opt/homebrew/bin/tectonic main.tex
elif command -v tectonic >/dev/null 2>&1; then
    tectonic main.tex
elif command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
else
    echo "Neither tectonic nor pdflatex found. Please compile main.tex with your preferred LaTeX distribution."
    exit 1
fi

echo "Compilation successful: report/main.pdf generated."
