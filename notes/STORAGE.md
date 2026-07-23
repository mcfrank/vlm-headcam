# Storage map — what's where (vlm-headcam)

Last updated 2026-07-23. Four tiers; the rule is **source in git, results in the book, caches on
local NVMe, cold/regenerable data on Oak.** Nothing is only in one place unless it's regenerable.

## Tiers

| Tier | Where | Holds |
|---|---|---|
| **git** | `github.com/mcfrank/vlm-headcam` (+ `~/Projects/vlm-headcam`) | all source (`src/`), the book (`book/`), figures, notes. The reproducible code. |
| **book** | `mcfrank.quarto.pub/learning-words-from-a-childs-eye-view` | all published results + figures. |
| **local NVMe** | `ccn2:/data2/mcfrank/vlm-headcam/` (~53 G) | hot working caches + model checkpoints + the Gemini annotations. |
| **Oak (cold)** | `oak-dtn:/oak/stanford/groups/mcfrank/vlm-headcam-archive/` | regenerable window caches + archived run checkpoints. |

Oak access: key-based, Duo-free from ccn2 (`~/.ssh/id_oak`, `oak-dtn` host block). See the
`train-on-ccn2` skill for the full Oak playbook.

## Local — `/data2/mcfrank/vlm-headcam/` (~53 G)

| Item | Size | Status |
|---|---|---|
| `scored/` (gemini_full.parquet, 1,145,371 rows; val sets) | 300 M | **KEEP — irreplaceable** (~$570 of Gemini calls to regenerate) |
| `runs/` (G_base_mil_full etc. — definitive checkpoints) | 3.1 G | **KEEP — reproducibility** |
| `manifests/` (grid_*, scale_*, cue_*, eval_frames_*) | 299 M | **KEEP** |
| `data/konkle/` (Konkle eval images) | 483 M | **KEEP — source images** |
| `emb_reg/` DINOv2 region cache (877k frames, CLS+4×4) | 27 G | regenerable: `embed_regions.py --frames grid_baseline_train.parquet --out emb_reg` (~2.5 h/GPU) |
| `emb_enc/` encoder train vectors (Ch 8, 5 encoders) | 8.0 G | regenerable: `assemble_encoder_cache.py` from Khai's `babyview_877k` |
| `emb_enc_grid_top/` topline grids (Ch 8, DINOv3-B/L) | 4.5 G | regenerable: same, from Khai's `_grid4x4` dirs |
| `emb_full`, `emb_reg_mp`, `emb_cls1`, `emb_reg_ho_0..7` | ~9 G | regenerable diagnostic/eval caches (mp/cls1 derive from `emb_reg`) |
| `emb_konkle*`, `emb_lev_vocab`, `eval_frames_mcfrank/`, `emb_crop_*`, `pose_*`, `lev_vocab_images/` | ~few G | eval caches + staged images; regenerable / re-stageable |

## Oak archive — `.../vlm-headcam-archive/` (~53 G, all regenerable)

| Item | Size | What |
|---|---|---|
| `emb_win_0..3` | ~36 G | ±2 s window region caches (frame-MIL). Freed locally 2026-07-23; here + regenerable. |
| `emb_win5_0..7` | ~14 G | ±5 s window region caches. |
| `home-runs/vlm_enc` | ~1.4 G | Ch 8 encoder-run checkpoints (Phase 1/1.5/2) |
| `home-runs/vlm_arch` | ~1.7 G | Ch 9 architecture + ladder-cleanup + diagnostic run checkpoints |

**Restore anything:** `rsync -rltP oak-dtn:/oak/stanford/groups/mcfrank/vlm-headcam-archive/<name> /data2/mcfrank/vlm-headcam/`

## Source data (Khai / lab, read-only — not ours to manage, but what we build from)

- `/ccn2a/dataset/babyview/2025.2/` — the release: `extracted_frames_1fps/`, `mp3/`, `outputs/` (transcripts, detections, **Gemini `scored/` is our copy**), `image_embeddings/babyview_877k/` (Khai's 5-encoder embeddings), `eval_{konkle,levante}_mcfrank/` (Khai's eval embeddings).
- `/ccn2/dataset/babyview/2025.2/outputs/pose_1fps*` — pose annotations.
