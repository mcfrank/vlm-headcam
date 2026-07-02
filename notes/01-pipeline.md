# Prototype pipeline (ccn2)

Env: `/data2/mcfrank/ladder/condaenv/bin/python` (torch 2.4.1, transformers 4.49, PIL;
no torchvision → DINOv2 uses the slow image processor, fine). Work dir `/data2/mcfrank/vlm-headcam/`.
Code in `src/` (deployed via scp from `~/Projects/vlm-headcam/src/`).

Frozen vision tower = DINOv2-base CLS (768-d), computed once and cached; every training
run is just a linear vision projection + a bag-of-words text tower on top (CVCL recipe).

## Steps

1. **Held-out eval videos** — a child reserved for eval only:
   `ls .../object_detections/cdi | grep '^S00360001_' > eval_videos.txt`
2. **Eval set** (`build_eval.py`): dominant-CDI-object frames from held-out detections →
   `manifests/eval_frames.parquet` (video_id, frame_idx, category). 7,288 frames / 61 cats.
3. **Training manifests** (`build_pairs.py`): from `full_clip_results.csv`, filter by
   `--min-clip` (the alignment knob), exclude the eval child/videos.
   - `smoke_aligned`  : `--min-clip 0.24`               → 137,179 pairs (mean clip 0.251)
   - `smoke_random`   : `--min-clip 0 --max-pairs 137179 --sample random` → size-matched control (mean 0.224)
4. **Embed** (`embed_frames.py`): union of all manifest+eval frames → `emb/` (resumable
   memmap `emb.f16.npy` + `index.parquet`). ~1.5 KB/frame.
5. **Train** (`train.py`): two-tower InfoNCE. Reports per epoch train loss, val
   contrastive loss (divergence check), and 4AFC accuracy (chance 25%).

## Experiment 1 — does alignment filtering rescue learning?

Size-matched contrast isolating alignment from quantity:
- aligned (clip>0.24) vs random (clip~0.22), both 137k pairs, same held-out eval.
Plus the field's failure mode: unfiltered full data (loss diverges) — to be added.
Prediction (Vong "absolute count" thesis): aligned > random on 4AFC; random ~ chance.
