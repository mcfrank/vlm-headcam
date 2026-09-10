# archive/ — deprecated project surfaces

Moved here 2026-09-10. Nothing in this directory feeds the paper.

- `book/` — the Quarto book *Learning words from a child's eye view* (2025.2 corpus; chapters on
  bootstrapping, cues, scaling, encoders, architectures). **Deprecated**: the paper supersedes
  it and its numbers predate the 2026.1 release, the audio-based English filter, and the final
  encoder set. The published copy at mcfrank.quarto.pub is a historical snapshot; do not
  rebuild or cite. Its diagnostics figures were ported to `figures/figS_corpus_*` from the
  committed `diagnostics/2026.1/` data.
- `supplement.qmd` — the 2026.1 data-diagnostics supplement and the methods stub (Gemini
  prompts, `[AI DRAFT]` methods text) used while drafting; the manuscript now carries the text.
- `eval/` — cue-titration era analysis scripts and outputs (book chapter 5, excluded).
- `enrollment_viz.R` — enrollment plot for the book's data chapter.

Legacy Python lives in `src/archive/` (with its own README); legacy runners in `runners/archive/`.
