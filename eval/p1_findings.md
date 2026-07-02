# P1 — self-bootstrapped alignment filter: results

Setup: warm up on ALL pairs (unweighted), then EM rounds that score each pair by the
model's own cross-modal cosine, fit a 2-Gaussian mixture, and reweight InfoNCE by the
"aligned" posterior. CLIP never used in training — only as a held-out yardstick. Eval =
4AFC on held-out child S00360001 (chance 25%). Grid: {within-kid S00510002, across-kid}
× {3 data sizes}, vs. unweighted baseline (plain) and CLIP-filtered oracle (topline).

Toplines (CLIP oracle): across 46.7 · within 34.9.

| pool | size | plain | boot | Δ(boot−plain) | ρ(weight, CLIP) | precision | base rate |
|------|------|-------|------|---------------|-----------------|-----------|-----------|
| across | 20k | 28.6 | 28.9 | +0.4 | 0.02 | 0.14 | 0.12 |
| across | 60k | 29.4 | 30.8 | +1.3 | 0.04 | 0.16 | 0.12 |
| across | 140k | 33.0 | 32.2 | −0.8 | 0.03 | 0.17 | 0.12 |
| within | 20k | 26.8 | 26.1 | −0.8 | 0.07 | 0.21 | 0.16 |
| within | 60k | 28.9 | 29.2 | +0.3 | 0.07 | 0.22 | 0.16 |
| within | 110k | 32.2 | 31.2 | −1.0 | 0.06 | 0.22 | 0.16 |

## Findings

1. **The naive bootstrap does not ignite.** boot ≈ plain everywhere (Δ within ±1.3, i.e.
   noise). The endogenous weight is essentially uncorrelated with true alignment
   (ρ 0.02–0.07), so its "selected" pairs are only marginally above base rate in precision
   → it effectively reweights at random and adds nothing over unweighted training.

2. **Data amount helps the *unweighted* baseline, not the bootstrap.** plain rises with
   size (across 28.6→33.0; within 26.8→32.2), but boot tracks it. More data does not
   create the cold-start toehold.

3. **Within vs. across dissociates, as predicted.** Within-kid gives a consistently
   *stronger* (if still weak) alignment toehold (ρ ~0.07 vs ~0.03; precision lift ~1.4×
   vs ~1.3×) — consistency aids the cold start. But within-kid's *ceiling* is far lower
   (oracle 34.9 vs 46.7) — one child's exemplars are fewer and less diverse, so meanings
   generalize worse to a held-out child. Easier toehold, lower ceiling; harder toehold,
   higher ceiling. Neither combination lets the naive bootstrap win.

## Why it failed (mechanism)

The cold-start requires that, after warming up on ~88% misaligned pairs, the model score
aligned pairs higher than misaligned ones. At BabyView's alignment floor the warmed-up
mapping is too weak to do this — the per-pair cosine is ~uncorrelated with true alignment
(ρ≈0.03). The memorization toehold that self-training relies on is real but *too faint*
at this noise level to bootstrap from instantaneous similarity. This is the risk flagged
up front: BabyView is far noisier (≈5–13% aligned) than the 20–50% noise regimes where
noisy-correspondence methods (NCR etc.) succeed.

## Implication

A stronger *endogenous* signal is needed than instantaneous cosine on a single (noisy,
midpoint) frame. Top candidate: **MIL over the utterance window (and object regions)** —
let the utterance match its best-matching frame/region instead of an arbitrary one. This
manufactures a cleaner positive per utterance, which should both raise achievable accuracy
and strengthen the toehold the EM needs. Co-teaching / learning-dynamics E-steps and an
easy-first curriculum are secondary options.
