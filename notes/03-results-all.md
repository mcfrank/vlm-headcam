# Results: experiments A–D (alignment filtering on BabyView)

All: two-tower InfoNCE, frozen DINOv2-base vision + bag-of-words text, 20 epochs.
Eval = CVCL-style 4AFC on CDI categories from YOLOE detections on **held-out children**
(chance 25%). Frozen-feature probe → training is a linear vision proj + word embeddings,
so this isolates the *alignment signal*, not optimization/capacity effects.

## B — Robustness (aligned vs size-matched random; 3 children × 3 seeds)

| held-out child | aligned (>0.24) | random (matched) | gap |
|----------------|-----------------|------------------|-----|
| S00360001      | 46.9 ± 0.2      | 32.6 ± 1.6       | +14.3 |
| S00240001      | 43.8 ± 0.6      | 33.6 ± 0.3       | +10.1 |
| S00370002      | 44.5 ± 0.7      | 32.4 ± 0.3       | +12.2 |
| **mean**       |                 |                  | **+12.2** |

The alignment gap is robust and tight. At identical data size (~137k pairs), filtering to
aligned pairs buys +12 points cross-child. Confirms the exp-1 finding with error bars.

## A — Brute scale vs filtering (unfiltered 1.14M, held-out S00360001)

best 4AFC: **aligned 137k = 46.7  >  unfiltered 1.14M = 43.2  >  random 137k = 34.6**

**8× more (unfiltered) data is WORSE than the filtered subset.** The aligned pairs are a
subset of the 1.14M; adding ~1M misaligned pairs *dilutes* the signal (46.7→43.2) but
doesn't destroy it (still > random, because the aligned subset is in there). Filtering
concentrates signal; scale alone dilutes it.

Note: val loss here is *stable* (5.39→5.13, never rises) — our small frozen-feature probe
can't overfit 1.14M pairs, so it does NOT reproduce the "loss diverges" failure reported
for end-to-end pixel training (Lin/EgoBabyVLM). That divergence was an optimization/
capacity phenomenon stacked on top of the alignment problem; here we see the alignment
effect cleanly on its own. Honest framing: our claim is "filtered > unfiltered," not
"unfiltered diverges."

## C — Count-vs-quality frontier (threshold sweep, held-out S00360001)

| min-clip | pairs   | best 4AFC   |
|----------|---------|-------------|
| 0.24     | 137,179 | 46.9 ± 0.2  |
| **0.26** | 21,877  | **48.4 ± 0.8** |
| 0.28     | 3,799   | 43.4 ± 0.7  |

Non-monotonic → there is a **sweet spot**. Raising quality (0.24→0.26) helps despite 6×
fewer pairs; pushing further (0.28, only 3.8k pairs) hurts — too little data. The optimum
(~22k clean pairs) is ~2.4× SAYCam-S's 9,320 aligned pairs. Directly engages Vong's
"absolute count" thesis: per-pair quality matters, but you still need enough pairs.

## D — Region grounding: object crops vs whole scenes (50k detection-video pairs)

best 4AFC:

|             | frame-eval | crop-eval |
|-------------|------------|-----------|
| frame-train | 39.2       | 45.4      |
| crop-train  | 42.9       | **48.7**  |

- **Crop-train > frame-train on the same eval** (+3.7 frame-eval, +3.3 crop-eval): pairing
  the utterance with the isolated object beats pairing with the cluttered scene.
- **Crop-eval > frame-eval**: isolating the object at test time also helps.
- Best cell (crop/crop = 48.7) is the highest in the study, beating whole-frame aligned
  (46.7) on only 50k pairs. Attacking clutter directly is a second real lever.
- Caveat: D's frame-train baseline (39.2) is below full aligned (46.7) because it's the
  50k detection-video subset using the detection frame (not the CLIP-best frame); the
  clean within-D comparison is crop vs frame on the identical 50k.

## Bottom line

Two independent, robust levers against BabyView's referential-noise problem:
1. **Alignment filtering** (+12 cross-child; filtered beats 8× unfiltered; sweet spot ~0.26).
2. **Region grounding** (object crops +3–4 over scenes; best overall 48.7).
Both are cheap and composable. Natural next step: combine them (crop-grounding on the
0.26-filtered set) and add real referent selection (attention/social cues) to D.
