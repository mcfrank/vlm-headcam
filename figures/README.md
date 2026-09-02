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
| — | **S1** region-MIL vs mean-pooled — *not built*, waiting on `F_*` no-MIL runs + R=1 caches |
| `figS2_diversity.py` | S2: diversity sweeps by pair budget (a null) |
| `figS3_alignment_encoders.py` | S3: the aligned arm for all four encoders |
| `figS4_tsne_encoders.py` | S4: the noun lexicon under all four encoders |
| `figS5_lexicon_encoders.py` | S5: relatedness by word class and encoder |
| `figS6_levante.py` | S6: the scaling experiment under the LEVANTE eval |
| `figS7_items.py` | *held out of the current ms* — item-level AoA + child-alignment across scale |
| `figS8_lexicon_alignment.py` | *held out of the current ms* — what alignment does to the lexicon |

S1–S6 are the manuscript's supplement, in order. S7/S8 build and are kept current but are
commented out in the ms; reinstating either means renumbering from its insertion point.

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
