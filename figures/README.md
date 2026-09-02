# figures/ — paper display items

One script per display item. **No script may hardcode a model number**: values come from
`results/` via `data.claim()` / `data.family()`, so a figure cannot silently drift from the
tables. Values whose source runs no longer exist are drawn **marked as provisional** (dashed/red
outline, plus a caption line) — see `notes/PROVENANCE.md`.

| Script | Display item |
|---|---|
| `fig1_pipeline.py` | design + pipeline (counts from `results/pipeline_counts.json`) |
| `fig2_scaling.py` | what moves the curve: encoders up (A), alignment left (B) |
| `fig3_development.py` | both evals in developmental time, against children |
| `fig4_lexicon.py` | lexicon: t-SNE, category structure, human relatedness |
| `figS1_alignment_encoders.py` | S: the aligned arm for all four encoders |
| `figS2_diversity.py` | S: diversity sweeps by pair budget (a null) |
| `figS3_levante_full.py` | S: Konkle beside LEVANTE, both scorings |
| `figS4_items.py` | S: item-level AoA + child-alignment across scale |
| `figS5_lexicon_encoders.py` | S: relatedness by word class and lexicon |

**Not yet buildable** (blocked on runs, see the session handoff): the region-MIL vs
mean-pooled comparison (no `F_*` no-MIL runs and no R=1 caches yet) and the t-SNE across
encoders (no lexicon cache for any `F_` family). `fig4` and `figS5` are still built from
PREVIEW-corpus lexicons (`C8_*`) for the same reason.

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
