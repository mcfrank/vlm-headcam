# Pre-submission cleanup plan — repo + CCN + Oak (revisited 2026-09-10)

Supersedes the GAPS list in `PAPER_REPRO.md` (2026-09-02) now that the experiments phase is
over: six encoders × the full family set (773 F_ runs), all controls, all evals, lexicon
analyses. Goal unchanged: **ALL and ONLY** — everything a paper number depends on is in git
(code) + results/ (numbers) + Oak (heavy artifacts), clearly separated from legacy, and nothing
irreplaceable lives in one place. Items marked **[Mike]** need a decision.

## 0. Current footprint (2026-09-10)

| where | what | size | copies |
|---|---|---|---|
| git | src/, figures/, results/ (86 files, 11 MB), notes/, human_check/ | — | GitHub |
| /data2/mcfrank/vlm-headcam | runs/ (773 F_ + 1,020 legacy dirs) | 19 G | **F_ NOT on Oak** (runs_20260829.tar predates every F_ family) |
| | emb_wf/, emb_wf_eval/ (no-MIL caches, 6 enc) | 16 G | regenerable in minutes (make_wf_caches.py) |
| | emb_ch8_eval/, emb_lev_* (eval caches, 6 enc) | ~5 G | eval_assets.tar on Oak predates vitl_bv/dinov3s |
| | scratch/lexicon (558 npz + text + w2v) | 3 G | mirrored to laptop `._lexicon_cache` (gitignored) — not on Oak |
| | scored/, manifests/ | 3 G | release copies canonical on ccn2b (retired originals) |
| | human_check/ (1,000 frames + sample) | 1 G | app data; frames = human subjects |
| /data2/mcfrank | frames_1fps_local | **616 G** | COPY of ccn2b frames (I/O speed) — retire |
| | emb_win5 + emb_win5b (window neighbor caches, 6 enc) | **637 G** | single copy; regenerable (~2 GPU-days total) |
| | c9_caches (vits_bv, vitb_bv, vitl_bv main region caches) | 99 G | **single copy**; ccn2b canonical dirs are symlinks into it |
| | dino_s2_vits / s3_vitb / s4_vitl (DINO session training runs) | 129 G | single copy; only ckpt 199999 matters |
| | hf_release/*.pt (native backbones, gitignored) | 0.4 G | single copy |
| /ccn2b …/image_embeddings | dinov3b, dinov3l, dinov3l_bv, dinov3s (real); vit*_bv (symlinks → c9) | — | dinov3s not on Oak |
| Oak `babyview-2026.1-mirror` | outputs/ (all layers, pose tar, 3 OTS/legacy embedding dirs), project/ (eval_assets, runs_20260829, runs_invalid tars) | ~195 G | — |
| Oak `vlm-headcam-archive` | 2025.2-era window caches + home-runs | ~53 G | legacy |

/data2 free: 167 G of 7 T. /ccn2b free: ~780 G.

## 1. Repo — code (ALL)

Add to the PAPER_REPRO chain (all committed since the audit):
- Stage 2 manifests: `build_aligned_controls.py`, `build_aligned_controls2.py`,
  `build_win_frames.py`, `build_win_frames2.py`, `build_win_frames_vitl.py`, `build_indomain_eval.py`
- Stage 2 runners: `run_final.sh`, `run_wf.sh`, `run_div30k.sh`, `run_controls.sh`,
  `run_controls2.sh`, `run_window.sh`, `run_window2.sh`, `run_vitl.sh`, `run_dinov3s.sh`,
  `scripts/{win5_embed,win5b_embed,vitl_embed_driver,dinov3s_driver,run_kchi_control}.sh`
- Stage 3: `eval_indomain.py`, `make_encoder_grid.py`, `make_experiments_table.py`,
  `make_methods_numbers.py`, `make_pipeline_counts.py`, `lex_rungs.py`; figures: `make_wordbank_40.py`
- Human check (SI): `human_check/` (sample.py, pull_frames.py, app/, analyze.py) — keep; must
  never contain frames, item→video maps, or rater files (check .gitignore covers
  `human_check/app/frames/`, `sample*.parquet`, responses).

ONLY: the legacy inventory in PAPER_REPRO stands. Decision **[Mike]**: (a) leave legacy in place
and let `PAPER_REPRO.md` + a `paper` git tag be the index (my recommendation — zero risk of
breaking the book), or (b) script a `paper-code/` export at submission. Either way, tag the
submission commit and mint a Zenodo DOI for the repo.

Housekeeping now: commit the `.gitignore` change (`release`, `hf_release/`); delete the
`results/*.pre6.csv` backups (the pre-fix versions are in git history); decide **[Mike]** where
`si.pdf`, `vlm_headcam_paper.pdf`, `talks/` live (the manuscript has its own repo; suggest
gitignore them here).

Figures: the figures session now has both numbered (`figS1_…figS7_`) and descriptive
(`figS_…`) scripts; README says descriptive names are canonical → delete the numbered
duplicates, and make `check_provenance.py` assert every `results/*` a figure reads is committed.

results/ hygiene: 86 files, several legacy (`lexicon_*_B26.csv`, `*_C8_*`, `lev_scaling.csv`,
`lexicon_partial_L-OTS.csv`, …). Generate `results/README.md` from the set of files actually
read by `figures/*.py` + `src/make_*.py` (paper-visible) and list the rest as legacy; don't
move files (the book reads some).

Methods numbers: regenerate `methods_numbers.tex` for six encoders (encoder table: 22M/86M/304M,
DINOv3 ViT-S/B/L ids, DINO fork commit hash for the BV three) and `\input` it in the paper.
Carry the manifest row-count note (1,686,392 rows / 4,133 vocab-empty / 1,682,259 effective).

Notes: keep `experiments.md` (journal), `CONTROLS_TABLE.md`, `PAPER_REPRO.md`, this file,
`HANDOFF_GEMINI_CHECK_APP.md`; move the session-to-session notes (`NOTE_TO_*.md`) to
`notes/sessions/` at submission; rewrite `STORAGE.md` (done today) and mark `DATA_LAYOUT.md`'s
target layout as achieved.

## 2. CCN node — what to retire, move, or archive

Ordered by payoff; nothing here is destructive until its Oak copy is verified (rsync -c).

1. **frames_1fps_local (616 G): delete.** Pure copy of `/ccn2b/…/extracted_frames_1fps` made for
   embedding I/O; all embedding is finished and the human-check frames are already pulled.
   Frees 616 G immediately. **[Mike] go?**
2. **c9_caches (99 G): move into the canonical tree.** The three BV main region caches are the
   real thing behind every BV number, and `/ccn2b/…/image_embeddings/{vits,vitb,vitl}_bv_grid4x4`
   are symlinks into node-local scratch. Move them to ccn2b (780 G free), replace the symlinks,
   re-run `verify_release.py`, then mirror all six encoder dirs to Oak
   (`outputs/image_embeddings/` currently holds only dinov3b, dinov3l, dinov3l_bv).
3. **runs/ F_ checkpoints (19 G): tar → Oak** as `project/runs_F_20260910.tar` (every paper number
   traces to these). Legacy prefixes (G/T/S1/P5/BXM/B26/C8, ~1,000 dirs) are book-era; B26/C8 are
   already in `runs_20260829.tar`; the rest: keep local (small) or add to the same tar **[Mike]**.
4. **DINO runs (129 G): keep ckpt 199999 of each + config + training_metrics; archive those to
   Oak (`project/dino_encoders_2026.1/`) alongside `hf_release/*.pt`; delete intermediate
   checkpoints** — coordinate with the DINO session (their tree). The released encoders are the
   scientific artifact; they also need a public home (see §4).
5. **Window neighbor caches (637 G) [Mike]**: (a) tar per encoder → Oak (bytes are fine: 10 T
   quota; ~12 tar files) then delete locally; or (b) delete outright (regenerable in ~2 GPU-days
   from the frame lists, which ARE committed). The window control is SI and done; I lean (a) so
   a reviewer request never costs two days, but it's 637 G of Oak.
6. **emb_wf (16 G)**: delete after the no-MIL SI figure is final (regenerable in minutes).
7. **eval caches + lexicon npz (8 G)**: refresh `project/eval_assets.tar` on Oak (now six
   encoders) and add `project/lexicon_emb_20260910.tar` (the 558 npz + word_pos + w2v).
8. **scratch, _retired_20260829, human_check**: `_retired_…` was retired because Oak holds it →
   delete after a last `rsync -c`; human_check stays until the rating study is analysed, then
   the frames dir goes (aggregate CSV lives in results/).

After 1–6: /data2 usage drops from ~6.8 T to ~5.4 T (our part from ~1.5 T to ~0.2 T).

## 3. Oak mirror — additions (babyview-2026.1-mirror/)

| add | size | why |
|---|---|---|
| outputs/image_embeddings/{dinov3s,vits_bv,vitb_bv,vitl_bv}_grid4x4 | ~130 G | only copies of the BV encoder caches; dinov3s missing |
| project/runs_F_20260910.tar | 19 G | every paper number |
| project/eval_assets_20260910.tar (refresh) | ~5 G | six-encoder eval caches |
| project/lexicon_emb_20260910.tar | 3 G | lexicon SI inputs |
| project/dino_encoders_2026.1/ (3 × ckpt 199999 + configs) + hf_release | ~10 G | the trained encoders |
| project/window_caches/ (optional, §2.5) | 637 G | temporal-window SI |
| project/human_check_20260910.tar (after the study) | 1 G | sample + frames + responses |

Inode-safe as written (a few hundred files). Verify with `rsync -c` before any local deletion;
update `MANIFEST.tsv` and the mirror README.

## 4. Paper-facing statements

- **Code**: GitHub repo at the `paper` tag + Zenodo DOI; the DINO fork (awwkl/dinov3 lineage)
  with commit hash; `PAPER_REPRO.md` as the map from every figure/number to its script.
- **Trained encoders**: native checkpoints (the HF export has the parity bug — ship native)
  via an OSF project or a gated HF org repo **[Mike]**; loader = the fork's `build_model` path.
- **Data**: BabyView 2026.1 access route (release team); our derived layers are part of the
  release tree (referent + language annotations, embeddings, pairs) — no frames leave.
- **Numbers**: `results/` is committed and cited by path in the SI; `results/encoder_grid.csv`
  is the encoder table's source; `methods_numbers.tex` macros for the text.

## 5. Order of operations

1. Now (no decisions needed): commit .gitignore; delete `.pre6` backups; regenerate
   methods_numbers for six encoders; results/README from figure reads; Oak additions §3 rows 1–4.
2. On Mike's go: delete frames_1fps_local; move c9_caches to ccn2b + fix symlinks + verify;
   window caches → Oak or delete; DINO intermediates (with the DINO session).
3. At submission: `paper` tag + DOI; notes/sessions move; results legacy list; encoder release.
