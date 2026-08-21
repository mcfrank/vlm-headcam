# Phases of the project

Written 2026-08-21, at the transition from exploration to a paper. Read this before trusting any
number: *which phase a result comes from determines what it means and whether it is comparable.*
Per-number status is in `results/provenance_report.csv`; the running detail is `notes/experiments.md`.

## Phase 1 — CLIP-filter prototype (2026-07-01 → 07-02) · superseded

First contact. Frozen DINOv2 + bag-of-words + InfoNCE on pairs filtered by a precomputed CLIP score
(`full_clip_results.csv`). Evaluation was a **CDI-detector 4AFC** built from YOLOE detections on
held-out children.

Established: alignment filtering beats size-matched random by ~+12 cross-child; 8× more unfiltered
data is *worse* than a filtered subset; region/crop grounding adds +3–4; an injected-label topline
(~72) showed big headroom over natural (~49).

**Why superseded:** the detector eval was noisy and child-specific (it scored *cat* at 14%), and
CLIP is a weak alignment signal. Numbers from this phase are **not comparable** to anything later.
Code: `archive/launchers/`, `src/build_pairs.py`, `src/train.py`, `eval/`. Runs: `A_*, B_*, C_*, D_*`.

## Phase 2 — bootstrapping + cues (2026-07-02 → 07-05) · negative results, still valid

Can a learner *discover* alignment without an oracle? Six mechanisms (cosine-EM, region-MIL E-step,
language prior, distinctiveness, cross-situational prototypes, curriculum) — none ignited
(ρ ≤ 0.1 between self-estimated and true alignment). Then: can an *accessible cue* do it? Language
(speaker, noun bias, discourse newness, prosody) and social/visual (hands, faces, gaze, pointing,
pose) cues all null; a **titration** with synthetic cues set the bar at ρ ≈ 0.3–0.5 used as a hard
gate, against a best real cue of ρ ≈ 0.14.

These conclusions survive the rig changes (they are null results and internal comparisons), but the
*absolute* accuracies quoted in this phase are old-rig. Runs: `P1_*, R_*, BX*, PF_*, T_*` (titration).

## Phase 3 — Gemini gold + Konkle eval (2026-07-03 → 07-06) · the eval that changed everything

Two pivots that make everything after comparable:
1. **Eval → Konkle object photos** (Vong's 60-category test split + 118 dev), out-of-corpus and
   leak-free, so we train on *all* children instead of holding one out. Worth ~+24 points over the
   detector eval — "the eval was the gap".
2. **Alignment gold → Gemini-2.5-Flash on Vertex**, scoring all 1,145,371 pairs for referential
   alignment 0–100 + a referent noun (`scored/gemini_full.parquet`). CLIP is kept only as a foil
   (the two correlate at ρ ≈ 0.14).

Also this phase: scaling + diversity sweeps (`G_sc_scale_*`), the ladder's first form, the
"what got learned" item analysis, and the LEVANTE-bench secondary eval. Still `train_region_mil.py`
with **post-hoc, final-epoch** evaluation — the **old rig**.

## Phase 4 — the clean rig + honest reframe (2026-07-14 → 07-15) · current standard

`train_frame_mil.py --window 0` became the one trainer: it evaluates on Konkle test-60 *during*
training and reports the **best** epoch. Re-running the ladder on it moved every rung up ~2–3 points
and, more importantly, corrected a real artifact: region-MIL's headline gain had been measured
against a weak **CLS** whole-frame baseline (52.9). Against the honest mean-pooled baseline (61.3)
the gain is **+4.0, not +9.7**. The book was rewritten around this: *free gains are modest; the
oracle rungs carry the climb* — which strengthens rather than weakens the thesis.

Also this phase: encoder comparison (ch8) and architecture variants (ch9) — both run on the clean
rig, both recovered from Oak.

**The provenance wound:** the ladder re-runs lived in `~/vlm_clean*` on ccn2 and were deleted in the
July home-dir cleanup; only `DONE` stubs survive. The ch4 ladder is currently **UNRECOVERABLE** and
must be re-run.

## Phase 5 — paper phase (2026-08-21 → ) · you are here

Freeze the pipeline, re-run what is unrecoverable, and build display items from committed tables
rather than transcribed constants. Infrastructure: `results/` (scraped tables + claims registry +
checker), `figures/` (paper display items, one script each), `notes/PROVENANCE.md` (what feeds what).
