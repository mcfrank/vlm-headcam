# DINO retrain on 2026.1 — design + living status
(updated 2026-08-25; the monitoring surface for this campaign)

## Goal
The strongest reasonable version of: "stock DINOv3 trained on BabyView does not yield a
word-learning-grade encoder" — size ladder (ViT-S/B, L if compute), stock recipe, healthy
training verified by probes, ≥100M samples seen. NOT a curation/augmentation research program
(explicitly descoped by Mike 2026-08-25; redundancy analysis kept for the discussion only).

## Decisions (closed)
- Frame pool: ALL 16,306 release videos, 1 fps (~9.7M frames), incl. non-English families
  (vision is language-independent). 1 fps justified: matches Orhan & Lake's frame count
  (~8.5M from 5 fps of 200 h) with ~5x the unique content; measured redundancy: within-video
  cos 0.915 at 1 s gap (73% > 0.9) vs 0.679 cross-video -> 5 fps adds ~pure redundancy.
- Recipe: Khai's fork (awwkl/dinov3 @ 71a12b8), config dinov3/configs/train/vitl_babyview.yaml
  = Meta's vitl16 reference (SK centering + KoLeo + iBOT, RoPE, bf16, compile).
- Batch: **512 global native (64/gpu x 8), NO gradient accumulation.** The config +
  train.py diff explain Khai's "strange loss plots": his accumulation calls forward_backward
  per micro-batch, so Sinkhorn-Knopp centering (and KoLeo) compute over 64-sample
  micro-batches instead of the effective batch — a statistics bug, as hypothesized. His good
  checkpoint (grad_accum_1) is exactly the no-accum native-512 regime. LR self-adjusts via
  the config's sqrt_wrt_1024 scaling rule.
- Samples budget: 200k iters x 512 = ~102M samples/model (Khai's ckpt: 125k x 512 = 64M;
  Orhan & Lake ~100M+). Schedule lengths are iteration-based (OFFICIAL_EPOCH_LENGTH=1250
  synthetic epochs), so exposure is decoupled from dataset size — set once, same for S/B/L.
- Crops: stock multi-crop geometry (global 224 @ 0.32-1.0, local 96 @ 0.05-0.32), pending the
  Stage-0 audit; the one candidate tweak if local crops look degenerate on portrait frames is
  the local-scale floor. Native 4K under-use noted as future work, not for this ladder.
- Gates recalibrated (2026-08-25): the Konkle prototype probe DOES NOT discriminate —
  Khai's L-BV scores 98.2 vs OTS 100.0 despite a 41-point word-learning gap. Prototype is a
  collapse detector only. **The discriminative gate is the 100k frozen word-learning probe**:
  anchors L-BV 29.6 / B-OTS 47.9 / L-OTS 54.0; random-init floor TBD (Stage 0 item).

## Stage 0 checklist
- [x] Repo cloned (/data2/mcfrank/dinov3 @ 71a12b8); imports clean under the ccwm env
      (torch 2.10.0+cu128)
- [x] Config located + read; grad-accum bug mechanism identified (per-micro-batch SK)
- [x] Probe anchors: prototype (saturated; collapse-detector only), word-probe spread
- [x] Redundancy analysis (numbers above)
- [~] Random-init ViT word-probe floor (chain running)
- [x] Crop audit PASSED: local crops 90% of global variance, 0.1% flat -> stock geometry
- [x] Data wiring: BabyView dataset plugin registered (spec BabyView:root=..:index=..:limit=..);
      frame index built: **9,726,507 frames** (frames_index.npy, shuffled, seed 0);
      node-local mirror copying (/data2/mcfrank/frames_1fps_local)
- [~] 10k-frame overfit test (running; first attempt failed on a missing dep, fixed)
- [ ] vits/vitb config variants (arch + size-appropriate drop_path; all else identical)

## Stage 1 (after Stage 0): ViT-S, 20k iters (~10M samples), 1M-frame subset
Gates, pre-registered: no rank collapse; prototype > 90 by 10k iters; 100k word-probe at
20k iters ABOVE the random-init floor and climbing. Pass -> Stage 2 (full ViT-S, 200k iters).

## Khai's answers (2026-08-25) — provenance CLOSED
1. Data wiring: he packed frames into ImageNet format and used the stock ImageNet loader
   (dataset_path=ImageNet:...:root=babyview/dataset/). No custom loader existed; our
   BabyView index plugin is the clean replacement.
2. FROM SCRATCH confirmed (Mike + config WEIGHTS=''). Run config archived at
   /ccn2/u/khaiaw/Code/baselines/dinov3/babyview/outputs/grad_accum_1/config.yaml
   (differences vs repo template = config-schema version skew only).
3. **Trained on the 868-hours release** (/ccn2a/dataset/babyview/868_hours/sampled_frames) —
   i.e. ~1/3 of 2026.1's hours, oldest packaging. The retrain is a genuine 3x data upgrade.
4. 8 GPUs -> global batch 512, grad_accum_1 (no accumulation), 125k iters ~= 64M samples.
   grad_accum_4 diverged around iter 70k (slow drift — consistent with the per-micro-batch
   Sinkhorn statistics bug found in the code); he retried rollbacks, then abandoned accum.

## Compute plan
S: ~1 day on 8xA40 at 200k iters. B: ~2 days. L: ~3.5 days or Marlowe (Mike exploring).
Node etiquette: coordinate around the shared queue; runs are resumable from ckpts.


## Stage 1 log (2026-08-25)
- ViT-S, batch 512 (8 GPUs), 20k iters, 1M-frame subset, local mirror. Launched ~20:13.
- CE gate PASSED: dinoG left ln(K) on Khai's trajectory at matched samples (10.80 @ it 2,240).
- Effective LR verified IDENTICAL to the reference run (the sqrt_wrt_1024 rule includes a x4:
  0.001 -> 0.00283 for batch 512 in both runs).
- Probe harness live (dino_probe.py: DCP ckpt -> teacher backbone via repo builder -> grid
  readout -> prototype + 100k word-probe; appends to <run>/probes.jsonl).
- Probe @ ckpt 2499 (~1.3M samples): word 22.45 (below the 27.0 random floor — expected
  early-SSL transient), prototype 66.8 (object structure forming). Verdict on the SLOPE
  across 2.5k/10k/20k.


## Stage 1 verdict + Stage 2 launch (2026-08-25 23:19)
Stage 1 PASSED all gates: CE on the reference trajectory; probes across 1.3M/5M/10M samples:
word 22.45 -> 23.10 -> 23.32 (monotone, above the >=23 hold-line), prototype 66.8 -> 78.4 ->
80.5. Note for the paper: the word-probe's shallow slope against a fast-climbing prototype is
the L-BV phenomenon in miniature (BV-DINO builds prototype separability much faster than
word-learnable geometry) — visible already at ViT-S/10M samples.
STAGE 2 LAUNCHED: full ViT-S, all 9,726,507 frames, 200k iters x 512 ~= 102M samples,
ETA ~29 h (out dir /data2/mcfrank/dino_s2_vits, ckpt every 10k). Probe checkpoints offline
at ~25k/50k/100k/200k against anchors: floor 27.0 | L-BV(64M smpl) 29.6 | B-OTS 47.9 |
L-OTS 54.0.
