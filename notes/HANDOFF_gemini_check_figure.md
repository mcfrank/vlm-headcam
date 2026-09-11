# Handoff to the figures session: human check of the Gemini alignment annotation

Written 2026-09-11 by the human-check session. Question for you: **does the manuscript need a
figure for this, or is a sentence + SI table enough?** My recommendation is at the end. All data
below is aggregate, already in the repo under `results/`; nothing needs the cluster.

## What was measured

1,500 (frame, utterance) pairs from the final training corpus, stratified by Gemini score (half
are Gemini zeros), rated blind by two lab members (V and S; labelled R1, R2) with a binary
question pinned to Gemini's 50-point anchor: *does the utterance refer to a concrete object
visible in this frame, even if small, partial, or one of many?* Each rater did all 1,500 at
~1.8 s/item. A third rater (Mike) did 391 under several instruction revisions; his ratings are
in the `_all` files only and should stay out of the manuscript. Strata carry corpus weights
(`n_pop/n_sample`), so every metric comes in a sample version (over the 1,500 items) and a
**corpus-weighted** version (what the number would be over all 1.31M eligible pairs). Design and
conventions: `human_check/README.md`; handoff rationale: `notes/HANDOFF_GEMINI_CHECK_APP.md`.

## Headline numbers (V + S; `results/gemini_human_check.csv`)

| | sample | 95% CI | corpus-weighted | 95% CI |
|---|---|---|---|---|
| inter-rater κ (1,487 shared items, 87.6% agreement) | 0.75 | | | |
| Gemini ≥50 vs. human consensus: precision | 0.88 | 0.85–0.90 | 0.88 | 0.85–0.90 |
| recall | 0.84 | 0.82–0.87 | 0.56 | 0.48–0.64 |
| F1 | 0.86 | 0.84–0.88 | 0.68 | 0.62–0.74 |
| agreement / κ | 0.87 / 0.74 | | | |

Consensus = both raters agree; the 185 items where they disagree are excluded from the
consensus rows (n = 1,315). Per-rater rows (`R1_binary_*`, `R2_binary_*`) use each rater alone
over all 1,500: R1 P/R/F1 = 0.78/0.80/0.79, R2 = 0.85/0.82/0.84, so Gemini agrees with each
rater (κ 0.61, 0.69) about as well as the raters agree with each other (κ 0.75).

Why weighted recall is low while sample recall is high: the corpus is 89% Gemini zeros, and
raters call 13% of the noun-containing zeros and 7% of the other zeros aligned; scaled to the
corpus those are many missed pairs. Gemini is conservative: what it keeps is right ~88% of the
time; it drops roughly half of what a human would keep. For a training filter that is the safe
direction, and worth saying in the text.

## The two curves (figure candidates)

**A. Calibration** — `results/gemini_human_check_calibration.csv`, one row per Gemini bin
(`0_noun`, `0_other`, `1-49`, `50-60`, `70-75`, `80`, `90-95`, `100`). Plot `yes_rate` (share of
ratings that said yes, both raters pooled) with `yes_lo`/`yes_hi` (Wilson 95% CI; `n_ratings`
≈ 300 per aligned bin, 892/597 for the zero bins). `yes_rate_R1`, `yes_rate_R2` are the per-rater
versions if you want two thin lines. Values: 0.16, 0.10, 0.63, 0.65, 0.81, 0.83, 0.96, 0.89.
Monotone from 50 up; the ≥50 rule sits at the point where humans go from ~1-in-8 yes to ~2-in-3
yes. (The `1-49` bin is 0.17% of the corpus, n = 50 items; fine to drop or grey out.)

**B. Threshold sweep** — `results/gemini_human_check_threshold.csv`, rows `who ∈
{consensus, R1, R2}` × `threshold ∈ {50,60,70,80,90,100}`; columns `precision`, `recall`, `f1`
and their `_lo`/`_hi` bootstrap CIs, the `w_` (corpus-weighted) versions, and `corpus_share`
(fraction of the corpus kept at that threshold: 11.2% at 50, 8.8% at 70, 6.8% at 80, 3.7% at
90, 2.0% at 100). Consensus: P/R/F1 = 0.88/0.84/0.86 at 50, 0.92/0.72/0.80 at 70,
0.94/0.54/0.68 at 80, 0.96/0.35/0.52 at 90. F1 peaks at 50, i.e. the paper's threshold is the
F1-optimal one on human judgment; precision keeps climbing with the threshold, recall falls
faster.

`*_all.csv` = same files with Mike as R3 (three-rater α = 0.73; pairwise κ 0.69–0.75).
`gemini_human_check_pairs*.csv` = the pairwise table.

## Recommendation

Main text: one sentence with weighted precision / recall / F1 and the inter-rater κ, pointing to
the SI. SI: the headline table plus **one two-panel figure**, A (calibration with CIs, ≥50
marked) and B (precision, recall, F1 vs threshold for the consensus, with corpus share as a
secondary annotation). Panel A is the one that earns its place: monotone calibration is the
visual argument that the score is meaningful rather than a coin flip above the cut. Panel B is
optional; the same information is in the table. If space is tight, A alone in the SI.

Style: same bin order as the CSV; treat `0_noun` / `0_other` as two bars at x = 0 (they are the
same score, split by whether the utterance contains a concrete object noun); log or categorical
x, not a linear 0–100 axis, since the bins are unequal.

Not in these files (ask if wanted): per-child agreement spread (37 children with ≥10 items,
agreement 0.60–0.95, sd 0.08), referent accuracy from the earlier three-level format (61
ratings, 72% lemma-level match), and the qualitative failure modes (people and pet names Gemini
resolves to a visible being; picture books).
