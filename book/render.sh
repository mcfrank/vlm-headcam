#!/usr/bin/env bash
# Render the book. Figures are prebuilt PNGs (src/make_figures.py on ccn2), so no
# Python execution is needed at render time.
set -euo pipefail
cd "$(dirname "$0")"
# Clean stale output first: Quarto doesn't remove HTML for deleted/renamed chapters on an
# incremental render, which leaves orphan pages in the sidebar/ToC (e.g. an old "Where next").
rm -rf _book .quarto
quarto render "$@"
