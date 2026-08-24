# Status: claims × data × models

Updated 2026-08-23. Read with `notes/PHASES.md` (what each era means) and
`results/provenance_report.csv` (per-number drift from the book).

## Current rig

DINOv3-B/16 off-the-shelf · dev-117 epoch selection, test-60 reported · OOV categories score
chance over all 60 · dropout off during eval · `--min-coverage 0.9` enforced · per-run `metrics.json`.

## Experiment status

| Claim / figure | Release | Data | Models | Status |
|---|---|---|---|---|
| **Ladder** (ch4, fig3) | 2025.2 | ✅ | ✅ 30 runs, both arms | **current rig** |
| **Scaling** (ch6, fig2) | 2025.2 | ✅ | ✅ 30 runs | current rig; 1.14M point refused (79% cache coverage) |
| **English-filter controls** | 2025.2 | ✅ | ✅ 18 runs | current rig — see below |
| **Encoders** (ch8, fig5) | 2025.2 | ✅ | ✅ 45 runs (Oak) | valid; re-scored for the dropout fix |
| **Architectures** (ch9) | 2025.2 | ✅ | ✅ 15 runs (Oak) | valid — `train_arch` already called `m.eval()` |
| **Layer sweep S1** (suppl.) | 2025.2 | ✅ | ✅ 63 runs | done: ZWM peaks mid-depth (+2.1), V-JEPA2 flat |
| **Cues** (ch5, fig4) | 2025.2 | ✅ | ❌ **OLD RIG** | not re-run; Phase-2 numbers |
| **Item analysis / confusion** (ch7) | 2025.2 | ✅ | ❌ **OLD RIG (DINOv2)** | needs re-run from a current model |
| **LEVANTE** (ch7, fig6) | 2025.2 | ✅ | ❌ **OLD RIG** | needs re-run |
| **Diversity sweep** (ch6) | 2025.2 | ✅ | ❌ **OLD RIG** | manifests also need the tie-break fix |
| **Aligned scaling arm** (ch6) | 2025.2 | ✅ | ❌ **OLD RIG** | manifests need the tie-break fix |
| **2026.1 scaling / diversity** | 2026.1 | partial | ❌ none | blocked on embeddings |

## Data status by release

| | 2025.2 | 2026.1 |
|---|---|---|
| children / videos | 36 / 8,566 | **51 / 16,306** |
| utterance–frame pairs | 1,145,371 | **1,838,134** |
| frames | ✅ | ✅ permanent `/ccn2b` path |
| Gemini alignment | ✅ | ✅ **100%** (1,838,061; 183,833 aligned ≥50) |
| DINOv3 grid embeddings | ✅ 877,802 frames | ⚠️ **2 shards of N** |
| Language annotation | 🔄 pass 1 of 2 | ⏳ queued |

2026.1 is a strict **superset** of 2025.2 (all 8,566 videos are among its 16,306), and shared
videos are byte-identical (verified: 25/25 frame counts, 75/75 md5s).

## English filtering — resolved

The by-child survey filter is **rejected**. All at a matched 790,519 pairs:

| | children | 4AFC |
|---|---|---|
| full corpus (904,812) | 36 | 72.55 |
| random pairs | 36 | 72.02 |
| 8 **random** children dropped | 28 | 71.82 |
| drop **S00240001 only** | 35 | 70.59 |
| survey ≥50% | 30 | 70.26 |
| survey ≥80% | 28 | **67.50** |

Quantity costs 0.5, arbitrary child loss another 0.2 — but the survey filter costs 4.3 more. One
bilingual child (S00240001: 61,002 pairs, 68% English by survey, ~90% by measurement) accounts
for 1.4 of it on its own. The survey correlates r = 0.47 with what is on tape.

**Replacement:** the video-level rule in `bv-annotations/language/` — drop a recording if <50% of
its language-decidable utterances are English (recordings with <20 decidable utterances exempt),
then drop non-English utterances within retained recordings. Keeps all children. Waiting on the
language annotation.

## Open decisions

1. **Consolidate everything on 2026.1?** It is a superset, fully Gemini-annotated, and has 51
   children rather than 36. Cost: ~8 GPU-h to finish DINOv3 embeddings, ~46k chunks × 2 for
   language, then re-run the ladder + scaling (~30 GPU-h). Benefit: one release, one Methods
   paragraph, a stronger diversity claim, and no "why does fig 2 use a different corpus than fig 3".
2. **The 1.14M scaling point** needs the held-out 20% embedded in DINOv3, or it stays absent.
3. **ch5 / ch7 re-runs** are the remaining old-rig results.
