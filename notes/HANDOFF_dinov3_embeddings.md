# Handoff: DINOv3 (and other encoder) embeddings for BabyView

*Written 2026-08-14. Covers the encoder-comparison work in book Ch 8 (`book/08-encoders.qmd`).*

## TL;DR

- **Who embedded:** Khai Long Aw ran every encoder over the **877,802 topline training frames** and the
  Konkle/LEVANTE eval images, and cached one `.npy` per frame. **We did not run the encoders ourselves** —
  we assembled his per-frame files into our stacked cache format and trained the contrastive head on top.
- **Two DINOv3 models:** off-the-shelf **DINOv3 ViT-B/16** (HF `facebook/dinov3-vitb16-pretrain-lvd1689m`,
  768-d) and Khai's **BabyView-trained DINOv3 ViT-L** (tag `awwkl/dinov3-vitl-babyview`, 303M, 1024-d).
- **Result:** off-the-shelf DINOv3-B is the best encoder we have tested (whole-frame 70.8 / region-MIL 72.6
  Konkle 4AFC); BabyView-trained DINOv3-L transfers *far* worse (41.0 / 41.7). Full numbers in Ch 8.

## Where the raw embeddings are (Khai's cache, ccn2 shared NFS)

Root: **`/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/`**

| Dir | What | Frames |
|---|---|---|
| `babyview_877k/` | training set — one subdir per *readout*, one `<image_id>.npy` per frame | 877,802 (pinned by `frame_ids_877802.txt` in the same dir) |
| `eval_konkle_mcfrank/` | Konkle eval images, all readouts + all layers + grid4x4 | 1,020 |
| `eval_levante_mcfrank/` | LEVANTE vocab eval images, same structure | 683 |
| `babyview/` (older) | **do not use** — a different 128,395-frame subset from the full 5.4M tree, only 16% overlap with our train set | — |

`image_id = {video_id}_{frame_idx:05d}`. Files are 1-D fp16 vectors, or `[16, D]` for `_grid4x4`.

### Readout subdirs (naming = HF-style model tag + suffix)

| Model | Dim | Readout dirs |
|---|---|---|
| DINOv3 ViT-B/16 off-the-shelf | 768 | `facebook_dinov3-vitb16-pretrain-lvd1689m` (CLS), `…_meanpatch`, `…_grid4x4` |
| DINOv3 ViT-L BabyView | 1024 | `awwkl_dinov3-vitl-babyview` (CLS), `…_meanpatch`, `…_grid4x4` |
| V-JEPA2 ViT-L BabyView | 1024 | `awwkl_vjepa2-vitl-fpc16-256-babyview-bs3072-e140_layer{0,4,8,12,16,20,23}` (+`_grid4x4` at layer23 on train; all layers on eval) |
| ZWM 170M BabyView | 768 | `awwkl_zwm-babyview-170m_layer{0,4,8,12,16,20,23}` (+`_grid4x4` at layer12 on train) |
| ZWM 1B BabyView | 1280 | `awwkl_zwm-babyview-1b_layer{0,8,16,24,32,40,47}` (+`_grid4x4` at layer24 on train) |

Readout semantics (from Khai): only the DINOs have a CLS token, so the bare `<tag>/` dir is CLS; V-JEPA2 and ZWM
have no pooler, so those are **mean-pooled patch tokens, layer-normed first**. `_meanpatch` exists for the DINOs
so all five compare apples-to-apples. Layers are 0-indexed block outputs; last layer taken after the final norm.
Recommended layers: ZWM → middle (170M: 12, 1B: 24), V-JEPA2 → last (23). `_grid4x4` = 16 spatial cells,
saved on train **only at each model's recommended layer** (all-layer grids would be ~650 GB); eval has all layers.

### Gotchas Khai flagged
- **Anisotropy:** mean-pooled patch features have cosine ~0.82–1.0 between unrelated images (vs ~0.03 for DINOv3
  CLS). Center features (subtract the mean) before RDMs / cosine analyses. Our trainer has `--center` for this;
  it made no difference to 4AFC for the DINOs.
- **Layer choice matters a lot for reconstruction-trained models** — ZWM-170M across-image cosine is U-shaped
  in depth (0.97 → 0.82 at layer16 → 0.95), so mid-depth is far more discriminative.
- ZWM-170M's train grid (layer12) and whole-frame vector are at *different* layers → its region-MIL row is a
  readout mismatch, not a real drop (Ch 8 footnote).

## Where the model weights are

- **DINOv3-B off-the-shelf:** Hugging Face `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated; accept the
  license on HF, then `AutoModel.from_pretrained`).
- **DINOv3-L BabyView, V-JEPA2, ZWM (Khai's):** the readout dir names are his HF-style tags (`awwkl/…`),
  which suggests they live under his HF account and/or his ccn2 tree. **I do not have the on-disk checkpoint
  paths or the embedding script recorded** — this is the one thing to get from Khai directly (see below).

## What we did with them (our side, ccn2 `/data2/mcfrank/vlm-headcam/`)

1. **Assemble** Khai's per-frame `.npy` into our stacked cache (`emb.f16.npy [N, R, D]` + `index.parquet`),
   in the row order of an image-ids file. Script: `src/assemble_encoder_cache.py` (32-thread reads to survive
   NFS small-file latency; ~minutes per readout).
   ```bash
   T=/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_877k
   python src/assemble_encoder_cache.py --base $T \
       --readout facebook_dinov3-vitb16-pretrain-lvd1689m_meanpatch \
       --ids manifests/topline_frame_ids.txt --out emb_enc/dinov3b_ots        # R=1 whole-frame
   python src/assemble_encoder_cache.py --base $T \
       --readout facebook_dinov3-vitb16-pretrain-lvd1689m_grid4x4 \
       --ids manifests/topline_frame_ids.txt --out emb_enc_grid_top/dinov3b_ots  # R=16 region grid
   # eval: same, --base .../eval_konkle_mcfrank --ids manifests/konkle_eval_ids.txt --out emb_enc_eval/<m>_konkle
   ```
   Assembled caches now on ccn2: `emb_enc/<model>` (whole-frame, 8 G total), `emb_enc_grid_top/{dinov3b_ots,
   dinov3l_bv}` (4.5 G), `emb_enc_eval/`, `emb_enc_grid_eval/`. All **regenerable** from Khai's cache
   (see `notes/STORAGE.md`).
2. **Train** the same contrastive head on each (`src/train_frame_mil.py --window 0` = the clean rig; R=1
   degenerates to whole-frame two-tower, R=16 = region-MIL; `--emb-dim` picks up D automatically):
   ```bash
   python src/train_frame_mil.py --window 0 \
       --manifest manifests/grid_baseline_train.parquet \
       --caches emb_enc/dinov3b_ots --eval-cache emb_enc_eval/dinov3b_ots_konkle \
       --eval-frames manifests/eval_frames_konkle.parquet --seed 0 --out runs/E_dinov3b_ots_wf_s0
   ```
   3 seeds each; runs archived on Oak at `vlm-headcam-archive/home-runs/vlm_enc/`. Figures via
   `src/make_encoder_ladder_fig.py`.
3. **Extended ladder** (Ch 8 §"extended ladder") reran the oracle rungs (alignment filter, referent label)
   on the DINOv3 grids the same way — same trainer, filtered manifests.

## How to embed more frames

There are two honest routes, depending on whose model:

- **Off-the-shelf DINOv3-B — we can do this ourselves.** Load `facebook/dinov3-vitb16-pretrain-lvd1689m`
  via HF transformers, run frames at 1 fps, and write per-frame `.npy` matching Khai's conventions
  (fp16; CLS → `<tag>/`, mean of patch tokens → `<tag>_meanpatch/`, patch tokens average-pooled to a 4×4
  grid → `<tag>_grid4x4/` as `[16, 768]`). Our existing DINOv2 embedder (`src/embed_regions.py`) is the
  template — it already does CLS + 4×4 grid pooling; swap the model id and the grid/dim. Then
  `assemble_encoder_cache.py` as above. Budget: ~2.5 GPU-hours per ~900k frames on an A40 for a ViT-B.
- **Khai's BabyView-trained models (DINOv3-L, V-JEPA2, ZWM) — ask Khai** for (a) the checkpoint paths on
  ccn2 / HF, and (b) his embedding script, so new frames match the existing readouts *exactly* (same
  layer indexing, layer-norm-before-mean, grid pooling). Do not re-derive these independently — a subtle
  readout difference (e.g. norm before vs after pooling) silently breaks comparability with the 877k cache.
  Note also his cache is keyed by `image_id` alone — Konkle and LEVANTE were split into two dirs precisely
  because 24 ids collide across sets; keep new eval sets in their own dir.

**Frame ids to embed** come from a manifest: the training set is pinned by `frame_ids_877802.txt` (=
`manifests/grid_baseline_train.parquet` unique frames); Konkle eval ids in `manifests/konkle_eval_ids.txt`.
The full 1 fps tree has 5,419,919 jpgs across 8,566 video dirs — the 877k is what the manifest references,
not what's on disk — so any expansion (e.g. the ±2 s / ±5 s window frames, or 2026.1) is a new id list.

## Open items
- Get Khai's checkpoint paths + embedding script into this repo (or a pointer), so the BabyView-encoder rows
  are reproducible without him.
- The BabyView-trained **DINOv2** from the original CCN paper (low-scaling result) was discussed as another
  comparison point but never located/embedded — still worth doing if that model can be found on ccn2.
- ZWM-170M whole-frame vs grid layer mismatch: a matched-layer rerun would clean up the one non-monotone cell.
