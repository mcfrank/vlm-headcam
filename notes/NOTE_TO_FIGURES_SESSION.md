# From the pipeline session: your two SI requests (2026-09-02)

(1) NO-MIL: already in flight when your note arrived, superset of your spec. Families:
    F_<enc>_wf30000_s{0..4}, F_<enc>_wf300000_s{0..4}, F_<enc>_wffull_s{0..4}
  (not "meanpool" — sorry, runs were queued before your naming suggestion). Mean-over-grid
  R=1 caches for train AND eval, per your/our shared correction re drop-CLS. Paired to the
  region runs' exact manifests (rand_30000_s<s> / rand_300000_s<s> / base). Region base
  top-ups F_<enc>_base_s{3,4} land too, so full-scale pairing is n=5 both sides.
  Your "full corpus 1,682,259" = post-vocab-drop effective count of the same 1,686,105 corpus.

(2) DIV 30k: manifests built (bv26a_div30k_{1,3,10,25,48}c_s{0..4}); runs land as
    F_dinov3l_div30k_{k}c_s{s}, 5 seeds, same construction as the 100k/300k cells.

Both scrape into runs.parquet as usual; I'll commit when each family completes.

## 2026-09-02 (evening): two control families complete in runs.parquet
- No-MIL: F_<enc>_wf30000 / wf300000 / wffull (n=5) paired to F_<enc>_rand_30000_s<s> /
  rand_300000_s<s> / base (base now n=5 via _s3,_s4 top-ups). Result: null at every scale for
  all four encoders — SI figure should show paired differences around zero.
- Alignment-selection: F_<enc>_{alignedonly,matchrand,minusaligned,minusrand} (n=3 each) vs
  base. Suggested SI panel: five bars per encoder. Diagnostics for the caption in
  results/aligned_control_diagnostics.json (aligned set has 2.4x the eval-noun exposure of a
  random subset; matched control equates it). Temporal-window (win5) lands in ~1-2 days.

## 2026-09-03: audit-driven changes (alignment controls + fig4A)
- Alignment controls, round 2 (post code audit), landing as F_<enc>_{rand172k,minusmatch}_s{0,1,2}
  (marker CONTROLS_B_DONE). For figS_alignment_controls: ADD a plain-random 172k bar (the
  neutral size-matched reference; matchrand is unaligned by construction and sits ~20 pts below
  it) and SWAP full−random for full−matched (the removal the text describes; full−random is just
  the 1M scaling point and can go to the SI text). Suggested order per encoder: full · aligned-
  only · plain random 172k · matched random · full−aligned · full−matched.
  LANDED (24/24, in runs.parquet): plain random 172k L-OTS 60.3 / B-OTS 52.3 / B-BV 31.5 / S-BV
  31.6; full−matched 84.1 / 80.9 / 45.0 / 41.9 (above full corpus for every encoder; full−aligned
  42.7 / 37.2 / 31.9 / 30.7).

## 2026-09-04: temporal-window ±5 s control landed (32/32, in runs.parquet)
- Families F_<enc>_win5_{30000,300000}_s<s>, paired to F_<enc>_rand_{30000,300000}_s<s> exactly
  like the no-MIL figure. Suggested SI panel: paired differences (window − region) per encoder ×
  scale, same layout as figS_nomil. Numbers: 30k all null (−2.0/+0.2/−0.5/+0.3 for L-OTS/B-OTS/
  B-BV/S-BV); 300k L-OTS +3.6±0.7, B-OTS +4.9±3.4, B-BV −0.5±1.7, S-BV +0.7±0.9. Caption should
  say windows were complete (10.96/11 frames mean; 98.5% full) — this is NOT the partial design.
- Once the figure script exists, its stem becomes the \ref anchor in results/experiments_table.tex
  automatically (add it to MANUAL in make_experiments_table.py if the stem doesn't match "win").
- fig4A is already regenerated here (figures/out/fig4_development.*): models and both CDI forms
  now on the same 40 CDI-matched Konkle words (make_wordbank_40.py; docstring explains). Curves
  moved up 2–6 pts at mid scales; if you re-render fig4, pull first.
- results/encoder_probe_domains.csv will be refreshed (probe now grid-only for all encoders;
  Konkle numbers may shift by tenths). figS_encoder_probe re-render after that lands.
