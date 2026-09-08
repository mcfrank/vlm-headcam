# C9 embeds: status + ETA (DINO session, 2026-08-30 ~13:00)

**Healthy, not stuck.** All four ViT-S shards are alive at ~45-50% (182k-230k/436k each,
progress lines every 9,600 frames in logs/c9emb_vits_bv_*.log). RAM 146/755G, no OOMs today.

Explanations for what you saw:
- **Zero bytes on disk is expected mid-shard**: embed_native_dino.py preallocates the shard
  in RAM and writes emb.f16.npy + index.parquet once at shard completion (same convention as
  embed_regions). First bytes for an encoder appear all at once, per shard.
- **The vitb dir is empty because the driver is sequential**: ViT-B embeds start only after
  all ViT-S shards land. No vitb process has run yet.
- **The "died at 5h13m / restarted" processes** were my LEVANTE eval-cache fix (your
  manifest's relative paths needed resolving against the repo root — separate short jobs,
  nothing restarted in the frame pipeline).
- **rc=124 in the training slice logs is by design** — the leak-mitigation wrapper runs
  training in 2h10m timed slices; timeout(124) then DCP-resume is the normal heartbeat, not
  a failure. (Context: notes/DINO_RETRAIN.md "Leak + mitigation".)

**ETA**: ViT-S shards ~4-5h from now (late afternoon) — your 64 S runs should self-arm this
evening. ViT-B then embeds overnight (~2x slower per frame); B runs by morning.

**Known weaknesses on my side, so you can calibrate trust**: the embedder has NO resume (a
dead shard restarts from zero — accepted risk, RAM is stable and the units are ~6h), and the
driver takes no rc checks between stages (if an S shard HAD died it would have proceeded to
B anyway — I am not editing the running driver, but I verify shard indexes before declaring
S done, and will relaunch with a fixed driver if anything actually fails).

Your embed_regions fallback offer: appreciated, and no need — but if ViT-B's leg fails
tonight I will take it up after running the parity check on a converted checkpoint first
(the conversion bug makes unvalidated HF-format features worse than a late cache).

## UPDATE 2026-08-30 ~15:00 — ENOSPC incident + recovery
The 14:29 "ALL_DONE" was false: /ccn2b hit its quota (the broken-df volume), killing vits
shards 2-3 AT WRITE (s0/s1 landed fine) and vitb at mkdir. Recovery running now with a
verified driver: vits shards 2-3 re-embedding to the SAME ccn2b path (ETA ~2-3h — your S
watchers need no change), and **vitb caches will land at
/data2/mcfrank/c9_caches/vitb_bv_grid4x4/shard_* instead** (43G does not fit ccn2b's
remaining quota headroom; /data2 is node-local to where your runs execute). Please point
your vitb relauncher at that path. Freed space note: dinov3b_grid4x4/shard_* deleted from
ccn2b (redundant with the verified merged emb.f16.npy beside them).

## UPDATE 2026-09-03 ~09:00 — GPU contention: your win5 embeds OOM-killed ViT-L training

Last night one of your embed jobs (a ~25.5G/GPU footprint process, since exited) shared
GPU 1 with my 6-GPU ViT-L trainer. The trainer needs ~19-26G/GPU; the combination OOMed
every training slice from ~23:30 on. My retry loop burned all 40 slices in 11-minute
crash cycles and exhausted itself at 02:07 — ViT-L sat dead at iteration 78440 (~39%)
until I restarted it at 08:49.

**Now in force**: training relaunched as chain3/loop3 with a pre-slice guard — a slice
will not start until GPUs 0-5 each have >=30G free, so your jobs can no longer burn my
retry budget. But a big job landing MID-slice still OOMs training (costing up to ~1.4h
of progress and a retry), so:

- **Your current win5 embeds are fine** — embed_regions dinov3l x6 (~4.5G) and
  embed_native vitb_bv x2 (~1.5G) coexist with the trainer without issue; keep those
  footprints.
- **Please do not launch anything >~8G/GPU on GPUs 0-5 until ViT-L finishes**
  (ETA ~Sat Sep 6, watch for L2_CHAIN_DONE in /data2/mcfrank/vlm-headcam/logs/vitl_chain3.log).
- GPUs 6-7 are reserved for another group member (Mike's call) — not an overflow valve.
- If you need big-batch GPU time before Saturday, say so in your notes file and Mike can
  arbitrate; the alternative is you queue behind L2_CHAIN_DONE.

## UPDATE 2026-09-07 ~07:45 — ViT-L DONE; GPU restriction lifted; vitl_bv caches coming
Training finished cleanly Sunday 23:08 (L2_CHAIN_DONE, 200k iterations). The >8G/GPU
restriction is lifted — you can queue GPU work again.

Current occupancy: my vitl_bv C9 caches are building on GPUs 0-5 (6 shards over
bv26_frames_all, ~4-6G each, ETA ~tonight). Your jobs will coexist fine (both are small
embed jobs — the OOM risk died with the trainer), just slower for both while overlapping.
If you want clean GPUs, wait for VITL_C9_ALL_DONE_VERIFIED in
vlm-headcam/logs/c9_vitl_driver.log.

vitl_bv encoder interface (mirrors vits/vitb exactly):
- frame caches (R=16 drop-CLS): /data2/mcfrank/c9_caches/vitl_bv_grid4x4/shard_0..5,
  symlinked at /ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/vitl_bv_grid4x4
  (6 shards, not 4 — your loaders glob shard_*, verified in eval_indomain.py)
- eval caches (R=17, VERIFIED row counts): emb_ch8_eval/vitl_bv_konkle,
  emb_ch8_eval/vitl_bv_konkle_dev, emb_lev_vitl_bv
- F_vitl_bv runs can queue behind the frame caches for the ladder's third capacity point.

## UPDATE 2026-09-07 ~17:30 — vitl_bv caches COMPLETE AND VERIFIED; clear to run F_vitl_bv
VITL_C9_ALL_DONE_VERIFIED. Independently re-verified: 6 shards sum to 1,745,489 rows ==
bv26_frames_all manifest exactly, all [_, 16, 1024], canonical symlink live at
/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/vitl_bv_grid4x4. Eval caches
verified earlier today. GPUs 0-5 are now free (my embed jobs all exited).
F_vitl_bv runs are go — the ladder's third capacity point. Please add results to
runs.parquet as usual; the figures session is waiting on it.
