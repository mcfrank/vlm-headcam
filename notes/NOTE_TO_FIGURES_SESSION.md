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
- 2026-09-07: extended to 1M and FULL (F_<enc>_win5_1000000_s{0,1,2} paired to rand_1000000_s<s>;
  F_<enc>_win5_full_s{0,1,2} paired to base_s{0,1,2}; all in runs.parquet). Paired Δ: 1M +1.6/
  +2.0/0.0/+2.2; full **+4.5/+2.5/−3.8/+0.7** (L-OTS/B-OTS/B-BV/S-BV). So the panel is 4 scales ×
  4 encoders; the story is "OTS encoders gain from temporal MIL, most at full scale; BV never do".
  Full L-OTS with window = 86.1 — worth a marker on fig2's scaling panel or a sentence, Mike's call.

## 2026-09-08: encoder grid becomes 3×2 — L-BV landed, S-OTS running
- New encoder tags in runs.parquet: `vitl_bv` (L-BV, 304M, DONE 113/113: the same families as the
  other four) and `dinov3s` (S-OTS, 22M, landing 09-09). S-BV STAYS (Mike: needed to show B is an
  improvement, since the BV size effect saturates at B: S 39.6 → B 44.1 → L 44.2).
- Mike's label decision: parameter counts, not ViT letters — OTS 22M / 86M / 304M and BV 22M / 86M /
  304M (psychologist readers). Encoding suggestion: hue = regime (blues OTS, oranges BV as in theme),
  marker = size (shared across regimes), one six-entry legend. The three BV curves nearly overlap —
  that's the result, but markers must disambiguate them. fig2, fig4, figS_alignment_encoders,
  figS_nomil, figS_alignment_controls, figS_indomain, figS_levante, figS_encoder_probe, figS_item_
  difficulty, figS_lexicon_* all enumerate encoders — add vitl_bv/dinov3s and the new labels.
- L-BV numbers, for sanity: rand 30k 26.7 / 100k 29.2 / 300k 34.9 / 1M 40.1 / full 44.2; aligned
  100k 45.1; ladder t2 52.4; controls and no-MIL/window all within seed noise of B-BV.

## 2026-09-09: LEVANTE file bug — re-render fig4B and figS_levante after the new file lands
- results/lev_scaling_final.csv had legacy 2025.2-corpus runs labeled plain "L-OTS" and "L-BV";
  below full scale they shared (encoder, N, seed) with the final F_ rows, so your groupby means
  pooled 2025.2 models into the L-OTS curve (and "L-BV" meant Khai's preview encoder). The
  regenerated file labels them "L-OTS (2025.2)" / "L-BV (preview, 2025.2)"; nothing to change in
  the figure code except adding the two new encoders — the collisions disappear by label. Draft
  LEVANTE numbers are playable-only accuracy; fig4B plots "fair". Both are in encoder_grid.csv.

## 2026-09-09 (midday): HANDOFF — six-encoder data are in place; what to render
Data (all committed, all local under results/):
- runs.parquet — F_<enc>_<cond>_s<seed> for enc ∈ {dinov3s, dinov3b, dinov3l, vits_bv, vitb_bv,
  vitl_bv}; best_acc = dev-selected test accuracy. S-OTS is complete EXCEPT the window family
  (F_dinov3s_win5_*, lands ~2026-09-10 early; rerun scrape_runs + pull runs.parquet then).
- item_eval_final.csv — item-level Konkle for all six (rand, lad, align families; test60 + dev117).
  NB: new encoders have base s0–4 in the lad family, old four have s0–2 only (top-ups queued).
- konkle_wg40_per_seed.csv, wordbank_anchors_40.csv — fig4A inputs, now six encoders.
- lev_scaling_final.csv — six encoders + legacy rows relabeled "L-OTS (2025.2)", "L-BV (preview,
  2025.2)" (exclude by label; nothing else changed — old rows reproduce exactly).
- indomain_eval.csv (word learning, all F scaling models, six encoders); encoder_probe_domains.csv
  (probe, six encoders, grid-only for all).
- encoder_grid.csv — the summary 3×2 table (Konkle scales, aligned, ladder, controls, no-MIL,
  window, LEVANTE fair, in-domain), with sd and n; make_encoder_grid.py regenerates it.
Labels (Mike's decision): parameter counts. tag → label: dinov3s "OTS-22M", dinov3b "OTS-86M",
  dinov3l "OTS-304M", vits_bv "BV-22M", vitb_bv "BV-86M", vitl_bv "BV-304M". Suggested encoding:
  hue = regime (blues OTS / oranges BV from theme), marker = size (o/s/^ for 22/86/304M) shared
  across regimes. The three BV curves nearly coincide — that IS the result; markers disambiguate.
Figures to touch: fig2 (scaling + aligned scaling: six curves), fig4 (A: six curves on 40 words;
  B: LEVANTE fair, six), figS_alignment_encoders, figS_alignment_controls (6 bars × 6 encoders:
  full · aligned-only · plain random 172k · matched random · full−aligned · full−matched),
  figS_nomil (paired Δ, six), NEW figS_window (paired Δ, 4 scales × six; S-OTS column pending),
  figS_indomain, figS_encoder_probe, figS_levante, figS_item_difficulty, figS_lexicon_* (lexicon
  extraction for the new encoders NOT yet run — say if you need it before I get to it).
2026-09-10 01:53: S-OTS window family landed — runs.parquet now has all 113 runs for all six
  encoders (F_dinov3s_win5_* included), so figS_window's OTS-22M column can fill in. Paired Δ:
  −0.5/+3.6/+5.2/+2.2. NB for figS_nomil: S-OTS at full is the one non-null cell (−4.6±1.4).
2026-09-10: lexicon files for the new encoders are in results/ — lexicon_ws_scaling_F-dinov3s
  / F-dinov3s-aligned / F-vitl_bv / F-vitl_bv-aligned, lexicon_partial_<same>, lexicon_tsne_F_
  {dinov3s,vitl_bv}_base_s0_NOUN, lexicon_category_structure_F_{dinov3s,vitl_bv}_base, and
  lexicon_rungs.csv now has six encoders. figS_lexicon_tsne / _relatedness / _alignment can go
  to six.
Text-facing numbers: results/encoder_grid.csv is the single source for the encoder table in the
  paper (params 22M/86M/304M; add DINOv3 ViT-S/16 "facebook/dinov3-vits16-pretrain-lvd1689m" and
  the DINO session's dino_s4_vitl to the encoder methods table).
- fig4A is already regenerated here (figures/out/fig4_development.*): models and both CDI forms
  now on the same 40 CDI-matched Konkle words (make_wordbank_40.py; docstring explains). Curves
  moved up 2–6 pts at mid scales; if you re-render fig4, pull first.
- results/encoder_probe_domains.csv will be refreshed (probe now grid-only for all encoders;
  Konkle numbers may shift by tenths). figS_encoder_probe re-render after that lands.
