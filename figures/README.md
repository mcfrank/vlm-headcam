# figures/ — paper display items

One script per display item. **No script may hardcode a model number**: values come from
`results/` via `data.claim()` / `data.family()`, so a figure cannot silently drift from the
tables. Values whose source runs no longer exist are drawn **marked as provisional** (dashed/red
outline, plus a caption line) — see `notes/PROVENANCE.md`.

| Script | Display item |
|---|---|
| `fig1_pipeline.py` | design + pipeline: corpus, referential annotation, two-tower learner (counts from `results/pipeline_counts.json`) |
| `fig2_encoders.py` | scaling across three encoders (B-OTS, L-OTS, L-BV) on the same draws |
| `fig3_alignment.py` | alignment is the factor: the aligned arm vs the unfiltered reference, ~10x data equivalence |
| `fig4_development.py` | both evals in developmental time: Wordbank CDI trajectories (A), measured LEVANTE children (B) |
| `fig5_lexicon.py` | interpretability: t-SNE lexicon, relatedness vs scale, CDI-category structure |
| `figS1_scaling_saycam.py` | S (provisional): the unfiltered+aligned curves with single-child SAYCam reference points |
| `figS2_representation.py` | S: encoder bars + the ladder across encoders |
| `figS3_levante.py` | S: LEVANTE vs generative VLMs + item frequency scatter |
| `figS4_items.py` | S: item-level AoA scatters + child-alignment across scale |
| `figS5_lev_scaling.py` | S: the scaling experiment under Konkle and LEVANTE side by side (pairs axis) |
| `figS6_diversity.py` | S: diversity sweeps at three budgets (largely a null) |

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
