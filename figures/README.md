# figures/ — paper display items

One script per display item. **No script may hardcode a model number**: values come from
`results/` via `data.claim()` / `data.family()`, so a figure cannot silently drift from the
tables. Values whose source runs no longer exist are drawn **marked as provisional** (dashed/red
outline, plus a caption line) — see `notes/PROVENANCE.md`.

| Script | Display item |
|---|---|
| `fig1_pipeline.py` | design + pipeline: corpus → midpoint pairing → Gemini annotation → frozen two-tower → 4AFC |
| `fig2_scaling.py` | scaling in two currencies: training pairs (A) and developmental time (B) |
| `fig3_ladder.py` | the alignment ladder, single panel |
| `fig4_cues.py` | cue information vs the ignition band (A) and the titration that sets the bar (B) |
| `fig5_representation.py` | encoder comparison (A) and the ladder across encoders (B) |

```bash
make -C figures            # build all -> figures/out/*.pdf (+ .png preview)
make -C figures fig2_scaling   # one item (targets are the full script stem)
```

No titles, captions or subtitles inside a figure — those go in the manuscript. Provenance and
rig caveats print to the build log (`NOTE fig…`) instead; provisional values are still marked in
the drawing (dashed / wine outline).

`assets/frames/` holds face-blurred frames (src/blur_faces.py, eyeballed) for fig1; `assets/konkle/`
four Konkle photos for the 4AFC icon. `results/literature.csv` holds published reference points.

Conventions: PNAS widths (`theme.W1/W15/W2` = 3.42 / 4.5 / 7.0 in), 6.5–8 pt type, PDF with
editable text (`pdf.fonttype 42`), panel letters via `theme.panel()`, project palette shared with
the book so all three surfaces read as one project.

## Known TODOs
- `fig2` mixes rigs on one axis (PROVENANCE D1) — re-run the scaling curve on the clean rig.
- `fig3`/`fig5` DINOv2 rows are provisional (D6) — re-run the ladder.
- `fig5` panel B rungs for the two DINOv3 encoders are still literals pending an extended-ladder scrape.
- `fig4` is Phase-2 rig; either state it in the caption (current) or re-run the titration.
- `fig1` camera inset: drop the BabyView site photo at `figures/assets/camera.png` and rebuild.
- `fig2` asymptote is fixed at the clean-label ceiling: a free asymptote is unidentified with 6 points (95 ± 48).
