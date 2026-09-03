# figures/ — paper display items

One script per display item. **No script may hardcode a model number**: values come from
`results/` via `data.claim()` / `data.family()`, so a figure cannot silently drift from the
tables. Values whose source runs no longer exist are drawn **marked as provisional** (dashed/red
outline, plus a caption line) — see `notes/PROVENANCE.md`.

| Script | Display item |
|---|---|
| `fig1_pipeline.py` | design + pipeline (counts from `results/pipeline_counts.json`) |
| `fig2_scaling.py` | what moves the curve: encoders up (A), alignment left (B) |
| `fig3_lexicon.py` | lexicon: t-SNE, category structure, human relatedness |
| `fig4_development.py` | both evals in developmental time, against children |

Supplementary items carry **descriptive names, not numbers** — the manuscript orders them
and links with `xr`, so renumbering here would only cause churn. Control ids refer to
`notes/CONTROLS_TABLE.md`.

| Script | Control | Display item |
|---|---|---|
| `figS_indomain.py` | C1 | word learning evaluated in domain, on held-out BabyView frames |
| `figS_encoder_probe.py` | C2 | encoder separability does not explain the lexical gap |
| `figS_kchi.py` | C3 | removing the child's own speech, amount-matched |
| `figS_diversity.py` | C4 | diversity sweeps by pair budget (a null) |
| `figS_nomil.py` | C5 | region-MIL vs a single mean-pooled frame (a null) |
| `figS_alignment_controls.py` | C6 | five selection arms per encoder |
| — | C7 | temporal ±5 s MIL — *not built*, embeddings still running |
| `figS_alignment_encoders.py` | | the aligned arm for all four encoders |
| `figS_levante.py` | | the scaling experiment under the LEVANTE eval |
| `figS_lexicon_tsne.py` | | the noun lexicon under all four encoders |
| `figS_lexicon_relatedness.py` | | relatedness by word class and encoder |
| `figS_lexicon_alignment.py` | | what referential selection does to the lexicon |
| `figS_item_difficulty.py` | | item-level AoA + child-alignment across scale |
| `figS_alignment_scores.py` | | distribution of Gemini referential-alignment ratings |
| `figS_corpus_effort.py` | | recording hours per child (from `diagnostics/2026.1/`) |
| `figS_corpus_age.py` | | age coverage and longitudinal span |
| `figS_speech_density.py` | | utterance rate per video and per child |

The last three are ported from `supplement.qmd` (its figs 2–4) onto the project theme; they
read the committed release diagnostics in `diagnostics/2026.1/`, not the cluster.

```bash
make -C figures            # build all -> figures/out/*.pdf (+ .png preview)
make -C figures fig2_scaling   # one item (targets are the full script stem)
```

No titles, captions or subtitles inside a figure — those go in the manuscript. Provenance and
rig caveats print to the build log (`NOTE fig…`) instead; provisional values are still marked in
the drawing (dashed / wine outline).

`assets/frames/` holds face-blurred frames (src/blur_faces.py, eyeballed) for fig1; `assets/konkle/`
four Konkle photos for the 4AFC icon; `assets/camera.png` the BabyView-site line drawing (CC-BY).
`results/literature.csv` holds published reference points; `results/wordbank_anchors.csv` the
Wordbank CDI child trajectories (rebuild with `Rscript figures/make_wordbank_anchors.R`).

Conventions: PNAS widths (`theme.W1/W15/W2` = 3.42 / 4.5 / 7.0 in), 6.5–8 pt type, PDF with
editable text (`pdf.fonttype 42`), panel letters via `theme.panel()`, project palette shared with
the book so all three surfaces read as one project.

## Known TODOs
- `fig3`/`fig5` DINOv2 rows are provisional (D6) — re-run the ladder.
- `fig5` panel B rungs for the two DINOv3 encoders are still literals pending an extended-ladder scrape.
- `fig4` is Phase-2 rig; either state it in the caption (current) or re-run the titration.
- `fig1` camera inset: drop the BabyView site photo at `figures/assets/camera.png` and rebuild.
- `fig2` now fits the B26 curve with a free asymptote (identified: 88.5 ± 3.3). The aligned arm
  is off panel A until `B26_lad*` families land — the script prints a NOTE when they appear.
