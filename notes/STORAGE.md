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
per-layer READMEs). Embeddings: all seven `image_embeddings/<enc>_grid4x4` dirs (six paper
encoders + legacy `dinov3l_bv`) are real directories on ccn2b (the three BV caches were moved
off node-local scratch on 2026-09-10) and all are mirrored to Oak.

## Node — `/data2/mcfrank/` (after the 2026-09-10 archive: 134 G, was ~1.5 T)

| Item | Size | Status |
|---|---|---|
| `vlm-headcam/runs/` — 773 `F_*` (paper) + ~1,000 legacy dirs | 19 G | on Oak: `project/runs_F_20260910.tar` (F_), `runs_20260829.tar` (B26/C8) |
| `vlm-headcam/emb_wf/`, `emb_wf_eval/` | 16 G | no-MIL mean caches; regenerable in minutes; not archived on purpose |
| `vlm-headcam/emb_ch8_eval/`, `emb_lev_*`, `manifests/`, `scored/` | ~8 G | on Oak: `project/eval_assets_20260910.tar`; release copies canonical for scored/manifests |
| `vlm-headcam/scratch/lexicon/` | 3 G | on Oak: `project/lexicon_emb_20260910.tar`; mirrored to the laptop's `._lexicon_cache` |
| `vlm-headcam/logs/`, `human_check/` | 4 G | run logs (keep until submission); rating-app frames (human subjects; cluster only) |
| `dino_s2_vits/`, `dino_s3_vitb/`, `dino_s4_vitl/` | ~15 G | ckpt 199999 + config + metrics only; on Oak: `project/dino_encoders_2026.1.tar` (with `hf_release/`) |
| `tmp/` | 27 G | DINO session's exported native/HF-converted checkpoints (their working area; duplicates of the above) |
| `ladder/` | 21 G | a different project (LM ladder) |
| `frames224/` | 14 G | 224-px frame cache for the joint-training SI pilot; regenerable |
| `oak_stage/` | <1 G | staging dir + archive logs (`node_archive_20260910.log`) |

Deleted 2026-09-10 after Oak verification: `frames_1fps_local` (616 G copy), `emb_win5`/`emb_win5b`
(637 G → `project/window_caches/window_<enc>_20260910.tar`), `c9_caches` (99 G → real dirs on
ccn2b + Oak), DINO intermediate checkpoints (~115 G), 2025.2-era caches (~90 G →
`project/legacy_caches_2025_2_20260910.tar`), DINO probe caches (~50 G →
`project/dino_probe_caches_20260910.tar`), `_retired_20260829`. /data2 (shared 7 T): 74% used,
1.8 T free.

## Oak — `babyview-2026.1-mirror/` (2026-09-10)

`outputs/`: MANIFEST.tsv, README.md, README_pose_1fps.md, annotations/{referent,language},
image_embeddings/{dinov3b,dinov3l,dinov3l_bv}_grid4x4, pose_1fps (tar, 19 G) + bbox CSV/parquet,
transcripts CSV (1.6 G), registry, release/exclusion ids, frame manifests, migration log.
`outputs/image_embeddings/`: now all six paper encoders (`dinov3s`, `dinov3b`, `dinov3l`,
`vits_bv`, `vitb_bv`, `vitl_bv`) plus the legacy `dinov3l_bv`, each verified byte-for-byte.
`project/`: eval_assets.tar (0.7 G, Aug), eval_assets_20260910.tar (2.3 G, six encoders),
runs_20260829.tar (9.6 G: B26/C8), runs_F_20260910.tar (10.6 G: all 773 paper runs),
runs_invalid_20260822.tar (1 G), lexicon_emb_20260910.tar (2.2 G), dino_encoders_2026.1.tar
(final checkpoints ×3 + configs + hf_release), legacy_caches_2025_2_20260910.tar,
dino_probe_caches_20260910.tar, window_caches/window_<enc>_20260910.tar ×6 (637 G).

**Restore anything:** `rsync -rltP oak-dtn:/oak/stanford/groups/mcfrank/babyview-2026.1-mirror/<path> <dest>`

## Evaluation assets — `/ccn2b/dataset/babyview/eval_assets/` (2026-09-18)

Konkle test-60 / dev-117 images, LEVANTE vocabulary items + images, and manifests with paths
relative to that root (README there; protocol in `EVAL.md`). Moved from the node working tree;
the working tree's `manifests/{konkle_manifest,eval_frames_konkle,eval_frames_konkle_dev,
lev_vocab_manifest}.parquet` and `lev_vocab_items.csv` are symlinks into it (pre-move copies in
`manifests/_pre_shared_20260918/`). Oak: `project/eval_assets_shared_20260918.tar`.

## Laptop — `~/Projects/vlm-headcam`

Working clone; `results/` (11 M) is the committed source for every figure; `._lexicon_cache`
(2.1 G, gitignored) mirrors the node's lexicon scratch for the local lexicon analyses.
