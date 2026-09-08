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


## Autonomous chain (2026-08-26, Mike offline for a few days)
- Stage 2 resumed from 20k after a host-RAM OOM (cache_dataset=true at 9.7M frames x 8 ranks;
  now false). Sentinel monitor armed. dinoG back on the reference curve.
- s2 ckpt-19999 probe (10M samples, full corpus): **word 26.0, prototype 91.6** vs
  Stage-1 subset run's 23.3/80.5 at matched exposure -> frame diversity helps BOTH probes
  (+2.7 word, +11 proto). At 10% of schedule this ViT-S is within 3.6 word-probe points of
  the fully-trained 868h ViT-L (29.6) — the 102M-sample run has a live chance to pass it,
  which would put numbers on "more/diverse developmental data helps, but how much".
- dino_chain.sh armed: on clean Stage-2 exit -> probes (100k, final) -> **Stage 3 ViT-B
  auto-launches** (same config, drop_path 0.2, ~2.5-3 days) -> final probe. All cluster-side.


## Leak + mitigation (2026-08-26 10:45)
Second host-RAM OOM at ~25k iterations after restart — cache_dataset was NOT the cause; a
~200G/h leak across the 80 dataloader workers kills the node ~3h in, reproducibly. Until
diagnosed (memlog_* now records free RAM every 5 min for the post-mortem), training runs in
**2h10m timed slices** via dino_train_loop.sh: timeout -> DCP auto-resume from the last 10k
checkpoint -> next slice. Worst case per slice-death: ~70 min of progress. The chain
(stage2 -> probes -> ViT-B -> probe) uses the loop for both stages; sentinel re-armed.
Resumed from 40k; ETA now ~27-30h for Stage 2 including restart overhead.


## ViT-L false start + corrected launch (2026-08-31 23:00)
The first 6-GPU launch failed by DESIGN FLAW, not recipe: at L speeds a 2h10m slice reaches
~4,700 iterations but the first checkpoint was at 10,000 — every slice resumed from zero
(Sisyphus). Compounded by a zombie second chain instance (half-dead ssh block) crash-looping
on the distributed port. Diagnosis silver lining: two full slices ran with NO host-OOM — the
leak scales with samples/sec, and L consumes 4x slower than S, so the fuse is ~12h not ~3h.
Corrected run (dino_l_loop.sh): single-instance guard; 5h30m slices; checkpoint every 2,500;
6 GPUs (0-5; 6-7 reserved for the group); batch 384 throughout, LR auto-scaled (ladder
footnote: S/B at 512). Observed 0.6 it/s -> ETA ~5 days (~2026-09-05). Sentinel re-armed.


## Root cause of the recurring "ssh died mid-block" (2026-09-01 00:05)
Every kill sequence of the form `ssh ccn2 'pkill -f "dino_train_loop"; <more commands>'`
was killing ITSELF: the pattern matches the ssh session's own bash cmdline, pkill takes out
the shell, and the rest of the block never runs — leaving orphan processes that later
masquerade as zombies/duplicate instances (tonight's L_LOOP_REFUSED, yesterday's port
crash-loop). Fix, now standard: bracket the last character of any pkill pattern
(`pkill -f "dino_l_loo[p]"`) so the pattern never matches a cmdline containing itself.
ViT-L relaunched clean under the v2 loop (single trainer, no refusal, iter 0 at 00:04,
~2.2k orphan iters sacrificed — pre-first-checkpoint). Sentinel v3 armed. ETA ~Sep 5.

## 2026-09-03: OOM crash-loop postmortem → loop v3 memory guard
Pipeline session's win5 embed jobs shared GPUs 0-5 with the ViT-L trainer; a ~25.5G
process on GPU 1 left <100M free and every slice OOMed ~11min in (compile, then first
big alloc). Loop v2 had no notion of "the GPU is occupied": it burned all 40 retry
slices in 2.5h (23:30-02:07) at iteration 78440 and exited EXHAUSTED. ~12h stall
(02:07-08:49) before restart.

Fixes (loop3/chain3, deployed 08:49):
- **Pre-slice memory guard**: a slice waits (5-min polls, logged to vitl_slices.log,
  not counted against the 40-slice cap) until GPUs 0-5 each have >=30G free.
- **chain3 probes only 99999/149999**: 49999 was probed before the crash, and its ckpt
  has rotated out (max_to_keep=8) — chain2's until-loop on a rotated ckpt would have
  blocked all later probes until train end. Lesson: never gate on a checkpoint that
  rotation can delete; gate on "ckpt exists OR iteration passed it".
- Contention protocol written to NOTE_TO_PIPELINE_SESSION.md: embeds <=~8G/GPU coexist
  fine; anything bigger queues behind L2_CHAIN_DONE or goes through Mike.
Residual risk: a big job landing MID-slice still OOMs that slice (~1.4h max lost work,
then the guard holds the retry until memory frees). Accepted.

## 2026-09-07/08: ViT-L COMPLETE + released — ladder finished
Training done Sun Sep 6 23:08 (L2_CHAIN_DONE, 200k iters, clean rc=0). Probe trajectory
50k/100k/150k/200k: word 27.2/26.2/26.2/26.1, proto 95.4/96.6/97.9/97.8 — word-probe flat
at this probe's scale (expected compression; S/B ended 31.0/30.6), proto healthy. The
capacity answer comes from the full-corpus F_vitl_bv runs (pipeline session; caches below).

Release: mcxfrank/babyview-dino-vitl16 (public, verified). Conversion cos 0.984 (bf16-RoPE
twin, milder than S/B's 0.91); functional gate 3-seed HF word-probe 29.1+-2.3 vs native
26.1 — statistical twin, same protocol as S/B. Native teacher backbone (1.21G) in repo.

C9 caches for vitl_bv: eval caches row-count-verified (konkle 1020 / dev 1431 / lev 681,
R=17 D=1024); frame caches 6 shards -> /data2/mcfrank/c9_caches/vitl_bv_grid4x4 with
canonical ccn2b symlink (driver c9_vitl.sh, marker VITL_C9_ALL_DONE_VERIFIED).
