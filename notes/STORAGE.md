# Storage map — what's where (vlm-headcam)

Last updated 2026-09-10 (end of the experiments phase). Rule: **source in git, numbers in
results/ (git), heavy artifacts on Oak, hot caches on node-local NVMe.** Nothing irreplaceable
lives in one place. The pre-submission actions are in `CLEANUP_PLAN.md`; this file is the map
as it stands today.

## Tiers

| Tier | Where | Holds |
|---|---|---|
| **git** | `github.com/mcfrank/vlm-headcam` (+ `~/Projects/vlm-headcam`) | all source (`src/`, `figures/`, runners, `human_check/`), every paper number (`results/`), notes, release docs |
| **release tree** | `/ccn2b/dataset/babyview/2026.1/` | frames (canonical), pose, transcripts, registry, release ids, `outputs/annotations/{referent,language}`, `outputs/image_embeddings/<enc>_grid4x4` |
| **node NVMe** | `ccn2-14:/data2/mcfrank/` | run checkpoints, eval + no-MIL caches, window neighbor caches, DINO training runs, the local frame copy, human-check app data |
| **Oak** | `oak-dtn:/oak/stanford/groups/mcfrank/babyview-2026.1-mirror/` | mirror of the release `outputs/` (incl. pose tar, three embedding dirs, annotations) + `project/` tars (eval assets, B26/C8 runs) |
| **Oak (legacy)** | `oak-dtn:/oak/stanford/groups/mcfrank/vlm-headcam-archive/` | 2025.2-era window caches, book-era run checkpoints |

Oak is transfer-only (rsync/sftp; no shell). Inode quota is the binding constraint: many-file
trees ship as single tars.

## Release tree — `/ccn2b/dataset/babyview/2026.1/outputs/` (canonical, mirrored to Oak)

Layers and keys: see `DATA_LAYOUT.md` (target layout achieved 2026-08-29; `MANIFEST.tsv` +
per-layer READMEs). Embeddings: `image_embeddings/{dinov3s,dinov3b,dinov3l,dinov3l_bv}_grid4x4`
are real directories; `{vits,vitb,vitl}_bv_grid4x4` are **symlinks into node-local
`/data2/mcfrank/c9_caches/`** (to be moved onto ccn2b — CLEANUP_PLAN §2.2). Only dinov3b,
dinov3l, dinov3l_bv are on Oak so far.

## Node — `/data2/mcfrank/` (2026-09-10)

| Item | Size | Status |
|---|---|---|
| `vlm-headcam/runs/` — 773 `F_*` (paper) + ~1,000 legacy dirs | 19 G | **F_ checkpoints are behind every paper number and are NOT yet on Oak** → tar to Oak |
| `vlm-headcam/scored/`, `manifests/` | 3 G | working copies; release copies canonical |
| `vlm-headcam/emb_ch8_eval/`, `emb_lev_*`, `emb_enc_grid_eval/`, `emb_dv3_konkle_dev16/` | ~5 G | eval caches, six encoders; Oak `eval_assets.tar` predates the two newest |
| `vlm-headcam/emb_wf/`, `emb_wf_eval/` | 16 G | no-MIL mean caches; regenerable in minutes (`make_wf_caches.py`) |
| `vlm-headcam/scratch/lexicon/` | 3 G | lexicon npz/text/w2v; mirrored to the laptop's gitignored `._lexicon_cache`; not on Oak |
| `vlm-headcam/human_check/` | 1 G | rating-app frames + sample (human subjects; stays on the cluster) |
| `emb_win5/`, `emb_win5b/` | 637 G | ±5 s window neighbor caches, six encoders; single copy; regenerable (~2 GPU-days) from committed frame lists |
| `c9_caches/` | 99 G | **the** BV encoder region caches (vits/vitb/vitl); single copy |
| `dino_s2_vits/`, `dino_s3_vitb/`, `dino_s4_vitl/` | 129 G | DINO session's training runs; ckpt 199999 = the released encoders |
| `hf_release/` | 0.4 G | native backbones (gitignored); single copy |
| `frames_1fps_local/` | 616 G | copy of the release frames for embedding I/O; retire |
| `_retired_20260829/` | 1 G | retired originals whose canonical copies are on the release tree / Oak |

/data2 is a shared 7 T volume; 167 G free today.

## Oak — `babyview-2026.1-mirror/` (2026-09-10)

`outputs/`: MANIFEST.tsv, README.md, README_pose_1fps.md, annotations/{referent,language},
image_embeddings/{dinov3b,dinov3l,dinov3l_bv}_grid4x4, pose_1fps (tar, 19 G) + bbox CSV/parquet,
transcripts CSV (1.6 G), registry, release/exclusion ids, frame manifests, migration log.
`project/`: eval_assets.tar (0.7 G), runs_20260829.tar (9.6 G: B26/C8 checkpoints),
runs_invalid_20260822.tar (1 G). Additions pending: CLEANUP_PLAN §3.

**Restore anything:** `rsync -rltP oak-dtn:/oak/stanford/groups/mcfrank/babyview-2026.1-mirror/<path> <dest>`

## Laptop — `~/Projects/vlm-headcam`

Working clone; `results/` (11 M) is the committed source for every figure; `._lexicon_cache`
(2.1 G, gitignored) mirrors the node's lexicon scratch for the local lexicon analyses.
