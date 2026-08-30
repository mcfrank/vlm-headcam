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
