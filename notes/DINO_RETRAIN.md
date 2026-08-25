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
- [ ] Random-init ViT word-probe floor
- [ ] Crop audit: dump stock multi-crop on portrait frames; blank-local-crop rate
- [ ] **Data wiring** — NOT in the repo (config still says ImageNet). Ask Khai how his run
      read frames; else build an index-file dataset over extracted_frames_1fps, with a
      node-local packed copy (/data2, ~600G, fits) to avoid 9.7M NFS reads/epoch
- [ ] 10k-frame overfit test (loss must fall; features must cluster)
- [ ] vits/vitb config variants (arch + size-appropriate drop_path; all else identical)

## Stage 1 (after Stage 0): ViT-S, 20k iters (~10M samples), 1M-frame subset
Gates, pre-registered: no rank collapse; prototype > 90 by 10k iters; 100k word-probe at
20k iters ABOVE the random-init floor and climbing. Pass -> Stage 2 (full ViT-S, 200k iters).

## Questions for Khai (blocking marked *)
1. *How did the good run load BabyView frames (dataset class/path/packing)? Not committed.
2. Config says WEIGHTS='' / pretrained_weights='' => FROM SCRATCH — but dataset_path is
   stale in the same file, so: confirm the run's actual overrides (need his launch command).
3. Which release/frame set did ckpt 119999 train on?
4. How many GPUs (=> true global batch for the good run)?

## Compute plan
S: ~1 day on 8xA40 at 200k iters. B: ~2 days. L: ~3.5 days or Marlowe (Mike exploring).
Node etiquette: coordinate around the shared queue; runs are resumable from ckpts.
