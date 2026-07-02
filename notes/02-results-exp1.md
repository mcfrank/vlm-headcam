# Experiment 1 — alignment filtering rescues word-object learning

Setup: two-tower InfoNCE (frozen DINOv2-base vision + bag-of-words text), 20 epochs,
seed 0. Train = 36 children's BabyView frame-utterance pairs; eval = CVCL-style 4AFC
over CDI categories built from YOLOE detections on a **fully held-out child (S00360001)**,
60 categories, chance 25%. The two arms are **size-matched (137k pairs each)** and differ
only in alignment, isolating alignment quality from data quantity.

## Result

| arm                | mean clip | 4AFC best | 4AFC final | val-loss behavior |
|--------------------|-----------|-----------|------------|-------------------|
| aligned (clip>0.24)| 0.251     | **0.467** | 0.456      | min 4.97 @ep6, gentle rise to 5.10 |
| random (clip~0.22) | 0.224     | 0.354     | 0.341      | min 5.10 @ep3, **rises to 5.27** (overfit) |

**~11-point 4AFC gap at identical data size.** Alignment filtering is a real lever —
consistent with Vong's "absolute count of aligned pairs" thesis, now shown to transfer
across children on BabyView.

Second signal, in miniature = the field's failure mode: the random arm's **val loss goes
up** (5.10→5.27) while train loss drops to 3.15 — it memorizes referentially-misaligned
pairs that don't generalize. The aligned arm generalizes (flat val loss, higher 4AFC).
Best 4AFC coincides with the val-loss minimum (~ep3-6) → early-stop there.

Per-category (aligned): strong plant .90, car .83, couch .81, room .77, plate .73,
sidewalk .73; weak cat .17, bathtub .20, pillow .22, sky .22 (diffuse/ambiguous referents,
matching CVCL's easy=tight-cluster / hard=diffuse finding).

## Caveats
- One seed, one held-out child, bag-of-words text. Need seed×child robustness.
- Filter (CLIP frame-text sim) and eval (YOLOE detectability) may share a "visually
  obvious object" bias. Mitigations: eval child is fully held out; text tower is learned
  from scratch (not CLIP), so 4AFC reflects learned mapping, not CLIP leakage. Still, the
  honest framing is *relative* (aligned vs matched-random), not the absolute number.
- Both arms beat chance: residual BabyView alignment (~0.22) is nonzero and eval nouns
  are common objects.

## Next
1. Add the field's true failure arm: **full unfiltered data** (~1.28M) — does val loss
   diverge harder? (needs ~1.1M more frame embeddings; cheap, ~1.5KB each.)
2. Seed × held-out-child robustness (≥3 each).
3. Threshold sweep (0.20/0.24/0.26) — is there a count/quality sweet spot?
4. Then experiment 2: region grounding via YOLOE crops (object, not scene).
