# vlm-headcam: synthesis & prototype design

Goal: a prototype infrastructure to iterate on learning word–object mappings from
BabyView (child egocentric video + ASR transcripts), and to beat the current
failure mode where naive CLIP-style training on BabyView diverges (loss goes up).

## The diagnosis (converges across all background sources)

1. **Vong et al. 2024 (Science, CVCL).** A minimal CLIP variant learns word→referent
   from a *single* child (SAYCam-S): 61.6% on Labeled-S 4AFC (chance 25), 34.7% on
   Konkle. The essential ingredient (ablations) is *consistent temporal co-occurrence*
   of frame & utterance; architecture barely matters.

2. **Vong & Lake 2026 (follow-up, arXiv 2507.14749).** Redid it across S/A/Y with
   Whisper transcripts. Key result for us: **average alignment is identical across
   children (mean CLIP sim ≈ 0.22 for all), but the number of *highly-aligned* pairs
   differs and tracks performance.** Count of pairs with CLIP frame-utterance sim > 0.24:
   S = 9,320 (7.6%), Y = 3,514 (4.9%), A = 3,223 (4.4%) → S > Y > A = accuracy order.
   Thesis: **the absolute count of high-quality aligned examples is the driver**, not
   average quality, transcription method, or architecture.

3. **Long et al. 2025 (BabyView dataset).** 868h released (2500h collected, 1600 on
   ccn2). Vision SSL on frames scales terribly (>1e7 hours extrapolated to match
   ImageNet); transcripts are fine for LM training. Vision is the hard modality.

4. **Lin et al. 2026 (EgoBabyVLM).** CLIP/LLaVA on BabyView ≈ chance on grounding.
   Cause = weak semantic alignment: JSD(true-pairs ‖ shuffled) = **0.012 for BabyView**
   vs 0.916 for COCO. BabyView sits at the *100%-shuffled* (random) alignment regime.
   Shuffling COCO reproduces the failure monotonically → alignment is causal.

5. **Frank symbolic work (bayesian-word-learning ch.8).** Hand-labeled 40 naming events:
   **only ~12% of caregiver CDI-noun mentions name a present, in-view object.** The rest
   are displaced/abstract/book-depicted/off-FOV. Filtering the transcript side helps a
   little but "the input is the limit." Their open lever was a model-side intent prior.

**One-line diagnosis:** the learning signal is dominated by referentially-misaligned
pairs; effective learning moments are rare (~5-12%), and naive contrastive training
drowns in the noise.

## The lever this prototype tests first

Vong's "absolute count" thesis + BabyView's scale = a concrete, cheap, falsifiable bet:

- SAYCam-S succeeded on **9,320** aligned pairs (>0.24).
- **BabyView already contains 165,770 utterances with a >0.24 max-frame CLIP score**
  (precomputed, `full_clip_results.csv`, 1.28M utterances / 37 children). That is
  **~18× SAYCam-S's aligned count.**
- Hypothesis: training on the *filtered aligned subset* should succeed (and stop the
  loss from diverging) where naive full-data training fails — because the problem was
  never too little data, it was too few aligned pairs swamped by noise.

If true, this is the whole game: use scale + a cheap alignment filter to manufacture
enough effective learning moments. If false (filtering doesn't rescue it), that itself
is a strong, publishable negative that points past whole-frame alignment toward
region/attention grounding.

## Assets already on ccn2 (2025.2 release)

Base: `/ccn2a/dataset/babyview/2025.2/`
- `extracted_frames_1fps/<video>/NNNNN.jpg` — 8,566 videos, 1fps frames.
- `outputs/merged_transcripts_parsed.csv` — 4.56M token rows, spaCy POS/lemma/dep
  (lets us filter to concrete NOUNs, kill homograph traps like can/watch/orange).
- `outputs/full_clip_results.csv` — **1.28M utterances** with clip_score_min/mean/max
  (frame-utterance CLIP sim). THE alignment filter, free.
- `outputs/object_detections/cdi/<video>/bounding_box_predictions.csv` — YOLOE detections
  prompted with the CDI label set, for **2,969 videos**. class_name is a CDI noun +
  bbox + confidence + masked_pixel_count. → ready-made labeled eval + region grounding.
- `outputs/image_embeddings/babyview/{facebook_dinov2-base, facebook_dinov3-vitb16}/`
  — 100k+ precomputed frame embeddings (.npy). SAYCam embeddings also present (dinov3).
- `data_subsets/{random_132hours, single_child}` — predefined training subsets.

Env: `/data2/mcfrank/ladder/condaenv/bin/python` (torch 2.4.1+cu124, transformers 4.49,
**no torchvision/timm/open_clip** — will need a venv or add them). 8×A40, /ccn2 has 4.9T free.

## Proposed infrastructure

1. **Eval harness** (foundation, build first). CDI-category 4AFC in the CVCL style:
   build labeled test frames from confident+large YOLOE detections (frame where CDI
   object X is present & dominant ⇒ label X); score by cosine(text-emb(X), frame-emb).
   Chance 25%. Held out by video/child. Anchor to Vong numbers where possible.
2. **Data prep**: emit frame–utterance pair manifests with tunable clip_score threshold
   + POS filter; train/val/test split by video.
3. **Training loop**: CLIP-style two-tower (DINO vision tower + small text encoder),
   symmetric InfoNCE, filter threshold as the central knob.
4. **Experiment 1**: filter-threshold sweep (none / 0.20 / 0.24 / 0.26) × does it stop
   divergence and beat chance on the eval. Direct test of the aligned-count thesis.
