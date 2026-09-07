# Paper reproducibility audit — ALL and ONLY (2026-09-02)

Question: does the repo contain ALL code needed to produce every named result in
vlm_headcam_paper, and ONLY that code clearly separated from legacy? Status: one gap fixed
in this audit (wf cache builder recovered from node scratch), several flagged below.

## THE CHAIN — everything a named paper result depends on

### Stage 0 · release + annotation layers (repo: vlm-headcam + bv-annotations)
| artifact | producer |
|---|---|
| release consolidation, keys, verification | `migrate_2026.py`, `pose_rekey.py`, `merge_emb_shards.py`, `verify_release.py`, `make_mp3_2026.py` |
| pairs (midpoint rule) | `build_pairs_2026.py` (+ `common.py`) |
| referent annotation | `gemini_align.py` |
| transcript language (2-pass) | bv-annotations: `annotate_language.py`, `agree_passes.py` |
| AUDIO language (authoritative) | bv-annotations: `annotate_audio_language.py` |
| English filter (video-level, audio) | bv-annotations: `build_english_filter_audio.py` |

### Stage 1 · embeddings
`embed_regions.py` (frames, drop-CLS 4×4), `embed_konkle.py` (eval sets, R=17),
`src/make_wf_caches.py` (mean-over-grid R=1 for the no-MIL SI comparison — RECOVERED this
audit from node scratch; was never committed).
**EXTERNAL**: BV-encoder training + native embedding = the DINO-retrain session's fork of
facebook/dinov3 (github.com/awwkl/dinov3 lineage) + its `embed_native_dino.py`. NOT in this
repo — must be cited with a commit hash in the paper's code statement. Teacher backbones are
in `hf_release/*.pt` (git-UNTRACKED, 429M — see gaps).

### Stage 2 · experiments (all F_ families)
`build_ladder_manifests.py`, `build_diversity2.py` → `run_final.sh`, `run_wf.sh`,
`run_div30k.sh` → `train_frame_mil.py` (+ `train_region_mil.py` for RegionMIL/eval) →
`scrape_runs.py` → `results/runs.parquet`. Joint-training SI pilot: `cache_frames224.py`,
`train_joint.py`, `run_joint_pilot.sh`.

### Stage 3 · evals + numbers
`eval_items_konkle.py`, `eval_lev_scaling.py`, `make_pipeline_counts.py`,
`make_methods_numbers.py`, `make_diagnostics.py`; lexicon SI inputs: `lex_extract.py`,
`lex_score.py`, `lex_word2vec.py`, `lex_partial.py`, `lex_category_structure.py`,
`lex_tsne.py`, `lex_rungs.py`, `lex_ws_scaling.py`.

### Stage 4 · figures
`figures/` (Makefile, data.py, fig1–fig4 + figS2–figS8, theme, scaling_fit,
make_levante_ages) reading ONLY committed `results/*`.

## ONLY — legacy inventory (everything else in src/, tagged)
- **book/2025.2 phases** (results live only in the quarto book, not the paper):
  `build_{grid_manifests,phase2,scaling_manifests,topline,within_child,window_frames,
  full_frames,gemini_arms,exp_manifests,jobs,region,eval,crop_eval,headnoun,word_prior}.py`,
  `train{,_arch,_boot,_caption,_perword,_regionprior,_wordweight}.py`, `embed_{frames,crops,
  dinov3_layers}.py`, `assemble_encoder_cache.py`, `merge_emb.py`, `reeval_saved.py`,
  `scrape_evals.py`, `grid_agg.py`, `make_{waterfall,ladder_figs,scaling_figs,arch_fig,
  confusion,item_analysis,item_plot,dev_figure,encoder_ladder_fig,disagree_frames,
  gemini_examples,cdi_categories}.py`, `plot_*.py`, `eval_{model,per_category}.py`,
  `data_stats.py`, `run_{phase5,layer_pilot}.sh`
- **cues (ch5 — EXCLUDED from the paper by decision)**: `build_{cue_manifest,combined_cues,
  speaker_manifests,utterance_cues,pose_*}.py`, `pose_*.py` (12 files), `cue_*.py`,
  `*_for_pairs.py`, `add_prosody.py`, `imu_camera_pitch.py`, `gen_point_overlays.py`,
  `characterize_persons.py`, `caregiver_close_for_pairs.py`, `run_titration*.sh`, `eval/` dir
- **superseded by this paper's final versions**: `build_pairs.py` (→ _2026),
  `filter_english.py` + `langid_*.py` (fastText/transcript era → audio),
  bv-annotations `build_english_filter.py` (transcript → audio version),
  `eval_lev_vocab.py`/`agg_lev_vocab.py`/`make_lev_vocab_fig.py` (→ eval_lev_scaling),
  `build_english_controls.py`, `run_{2026_data,2026_pipeline,2026_studies,ch8,div2,
  english_controls,english_followup,scaling_enc}.sh` (preview-corpus runners → run_final)
- **infra, keep**: `blur_faces.py`, `make_figures.py`, `check_provenance.py`, `status.sh`
- **not code**: `skeletons/`, `scratch_lev/`, `vlm_disagree_review/`, `talks/`, `archive/`

## GAPS / TODO before submission
1. ~~wf cache builder uncommitted~~ → committed as `src/make_wf_caches.py` this audit.
2. **DINO training code + checkpoints**: fork+hash citation needed; `hf_release/*.pt` (429M)
   are untracked and NOT in the Oak mirror (created after) — add to Oak, and decide
   HF-vs-supplement distribution (HF export has the known parity bug — ship native).
3. **figures/fig1 still reads `results/corpus.csv`** (2025.2-era values) alongside
   pipeline_counts.json — figures session should confirm which panel values remain
   corpus-sourced and migrate or phase-tag them. `results/titration.csv` (cues-era) is
   also referenced in figures/data.py — confirm nothing paper-visible uses it.
4. **supporting_info.tex is still the PNAS template** (frog/example-image) — the figS2–S8
   includes need wiring.
5. `methods_numbers.tex` exists but is not yet `\input` in the paper — recommended.
6. Whisper version/pipeline line in Methods still needs the release-team detail.
7. L-BV (Khai's) numbers in any figure carry the released-artifact caveat until resolved.
8. Legacy files carry no in-file marker; this document is the authority. If a stricter
   "ONLY" is wanted for an archival code release, export the Stage 0–4 list to a clean
   `paper-code/` snapshot at submission (script it; do not move files mid-project).

## 2026-09-07 addendum: training-manifest row count
`manifests/bv26a_base.parquet` has 1,686,392 rows vs 1,686,105 English-filtered pairs: the Gemini
merge in `build_ladder_manifests.py` fans out 287 rows where (video_id, frame_idx, text) repeats in
both inputs. 4,133 rows are vocab-empty → 1,682,259 effective (unchanged). 0.02%; documented, not
rebuilt (rebuilding would invalidate every F_ run's exact manifest). Methods text should quote
1,686,392 training rows / 4,133 vocab-empty / 1,682,259 effective, with the 287-row note.
