# Supplementary materials — running list

Format: `src/make_diagnostics.py` runs **on ccn2** and emits a small, face-free aggregates
bundle (`diagnostics/<release>/*.parquet`, ~0.5 MB) that IS committed; `supplement.qmd` renders
locally against that snapshot. Raw data are restricted and 2.6 GB+ per layer, so the qmd cannot
render against them directly — but numbers still come from data rather than being typed in, and
`provenance.json` ties every number to a release + extraction date + row counts.

Re-run the extractor when data change; re-render; numbers update.

## S1. Dataset description
- [ ] Release composition: videos, children, hours, age range (2026.1: 16,306 / 51 / 2,701 h / 0.24–4.51 y)
- [ ] Per-child table: videos, hours, age range, camera type, survey % English
- [ ] Age distribution of recorded hours (histogram, and per-child spaghetti over age)
- [ ] Recording density: hours per child per month; gaps in longitudinal coverage
- [ ] What 2026.1 adds over 2025.2 (8,566 shared videos, 11,735 new)

## S2. Annotation layers — coverage and provenance
- [ ] Coverage matrix: for each layer (frames, transcripts, pose, referent, language,
      embeddings), % of the 16,306 release videos covered. **The single most important check** —
      catches missing shards and partial runs.
- [ ] Per-layer README contents (method, model, env pins, join key, date)
- [ ] Known gaps: 12 sub-second clips, 154 frameless pairs, 3,995 non-release videos

## S3. Language annotation  ← a standalone resource for other BabyView users
- [ ] Method: Gemini-2.5-Flash on Vertex AI, two independent passes, chunked context
- [ ] Reliability: 2025.2 = 99.22% pass agreement, 98.78% English among agreed,
      0.15% mixed, 0.92% undecidable
- [ ] Per-child table: **survey % English vs measured % English** (these disagree sharply —
      children logged "0% English" are 96–98% English on tape; r = 0.47)
- [ ] Video-level filter decisions and the ≥50%-English threshold
- [ ] Documented biases (see `bv-annotations/language/README.md`)
- [ ] Why not fastText: 99% precise on English but only 58% on non-English here (43% of
      utterances are 1–2 words)

## S4. Referent annotation
- [ ] Method: Gemini alignment 0–100 + referent word over 1.84M utterance–frame pairs
- [ ] Alignment score distribution, overall and per child
- [ ] % of pairs with a spoken referent vs referent visible but unnamed (~37%)
- [ ] Validation against CLIP and against human judgement

## S5. Pose annotation
- [ ] Method: YOLO12x → mmpose RTMW-x, 133-kpt COCO-WholeBody
- [ ] Persons per frame distribution; % frames with ≥1 person, by child and by age
- [ ] The gcp-name ↔ rec-id crosswalk (see `notes/DATA_LAYOUT.md`)

## S6. Model / results methodology
- [ ] Konkle eval: test-60 (headline) vs dev-117 (selection); **verified disjoint** —
      0 shared categories, 0 shared images
- [ ] Epoch selection on dev, reporting on test; note that selection noise at small N
      slightly flattens the scaling curve (conservative for the "still rising" claim)
- [ ] Scaling error bars: independent subsample per seed (5 draws for N ≤ 100k, 3 above),
      so spread captures subsample + init variance
- [ ] Diversity sweep: random draw of k children per seed (not the k largest)
- [ ] Region-MIL (R=16, max over regions) throughout; cache conventions R=1/16/17
- [ ] `--min-coverage` guard and per-run `metrics.json`

## S7. Reproducibility
- [ ] `notes/PHASES.md` — a number is only comparable within a phase
- [ ] `notes/PROVENANCE.md` — figure → script → data, and divergences D1–D8
- [ ] `results/published.csv` — claims registry, live values vs printed values
