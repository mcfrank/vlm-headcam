# Handoff: DINO campaign → final-rerun session
(2026-08-29, from the DINO-retrain session. Read with notes/DINO_RETRAIN.md for full logs.)

## What now exists (new since the audio-language split)
Two from-scratch DINOv3 encoders trained on ALL of 2026.1 (9,726,507 frames at 1 fps, 51
children, language-unfiltered by design — vision is language-independent), 102M samples each,
stock Meta recipe at batch 512, training verified healthy against Khai's reference run:
- /data2/mcfrank/dino_s2_vits  (ViT-S, final ckpt 199999)
- /data2/mcfrank/dino_s3_vitb  (ViT-B, final ckpt 199999)
Checkpoints are torch DCP; load the TEACHER backbone via /data2/mcfrank/dinov3/dino_probe.py's
pattern (repo builder + config.yaml from the run dir). 100k-probe anchors:
  random-init 27.0 | ViT-S 31.0 | ViT-B 30.6 | L-BV(Khai,868h) 29.6 | B-OTS 47.9 | L-OTS 54.0

## Headline for the paper (affects your framing, not your pipelines)
Capacity is FLAT for developmental encoders: S(22M)=31.0 ≈ B(86M)=30.6 across 102M samples,
with Khai's L(300M) at 29.6. The 17+ point gap to internet pretraining is not an architecture
artifact in either direction. ViT-L retrain likely unnecessary (pending Mike's call).

## What this means for the final rerun
1. **Headline rig stays L-OTS** (dinov3l_grid4x4 caches, already on ccn2b) — nothing about
   your English-filter/audio-based manifest work changes.
2. **Encoder ladder arms**: full-corpus word-learning evals for the two new encoders are the
   remaining piece — embed 1.75M pair-frames with each, then base/300k/100k x 3 seeds, same
   trainer/eval as C8_*. I am queueing the embeds tonight (node is idle); runs land as
   families C9_vits_bv / C9_vitb_bv in runs.parquet via the usual scraper. If you launch the
   final rerun before these finish, no conflict — they use the standard resumable pattern and
   share GPUs politely.
3. **CAUTION on L-BV numbers**: Khai's HF release may differ from his trained model (his HF
   config has 4 register tokens vs n_storage_tokens=0 in training; conversion possibly added
   untrained parameters). All existing L-BV numbers (C8_dinov3l_bv_*, 41.7-era E_* runs)
   measure the RELEASED artifact. Mike has questions in to Khai; hold off hard-coding L-BV
   values into text until that resolves. Our S/B numbers have no such issue (native weights).
4. **HF sharing is paused** on a conversion-parity bug (exported model loses ~1.6 word-probe
   pts vs native; see DINO_RETRAIN.md). Do not use /data2/mcfrank/hf_export/* features for
   any analysis — native-checkpoint features only.

## Shared-resource notes
- Node14 GPUs: free as of tonight except the C9 embed jobs I'm launching (4+4 GPUs, ~6-8h).
- /data2 holds the frame mirror (614G) + DINO run dirs (~50G) — do not clean these.
- The English-filter → manifest rebuild remains YOURS; when your audio-based filter lands,
  the bv26 manifests regenerate and everything downstream (including my C9 arms) should be
  re-run on the new manifests — flag me and I'll rerun the encoder arms on your final corpus.

Questions: ping this session or Mike. — DINO session
