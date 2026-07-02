#!/usr/bin/env bash
# Render the book. Figures are prebuilt PNGs (src/make_figures.py on ccn2), so no
# Python execution is needed at render time.
set -euo pipefail
cd "$(dirname "$0")"
quarto render "$@"
