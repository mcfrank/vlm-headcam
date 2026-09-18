# src/archive — legacy code (not used by the paper)

Moved here 2026-09-10 at the end of the experiments phase. **Nothing in this directory is
needed to reproduce any number in the paper**; the paper chain is `src/` proper (see
`notes/PAPER_REPRO.md` for the map from every result to its script). These files are kept
for the record: they produced the 2025.2-era book chapters (`archive/book/`) and the
excluded cue analyses. They are not maintained and mostly assume `src/` on `sys.path`
(`PYTHONPATH=src python src/archive/<file>.py`); expect stale paths.

## By era / purpose

**Book phases on the 2025.2 corpus (chapters 3–9 of `archive/book/`)**
manifests: `build_grid_manifests`, `build_phase2`, `build_scaling_manifests`, `build_topline`,
`build_within_child`, `build_window_frames`, `build_full_frames`, `build_gemini_arms`,
`build_exp_manifests`, `build_jobs`, `build_region`, `build_eval` (YOLOE eval set),
`build_crop_eval`, `build_headnoun`, `build_word_prior`. (`build_lev_vocab` was archived here
by mistake on 2026-09-10 and moved back to `src/` on 2026-09-18: it builds the paper's LEVANTE
item table.)
training variants: `train_arch`, `train_caption`, `train_perword`, `train_regionprior`,
`train_wordweight` (the two-tower baselines `train.py` / `train_boot.py` stay in `src/`
because the paper's trainer imports them).
embeddings: `embed_frames`, `embed_crops`, `embed_dinov3_layers`, `assemble_encoder_cache`,
`merge_emb`.
evaluation / aggregation: `reeval_saved`, `scrape_evals`, `grid_agg`, `eval_model`,
`eval_per_category`, `eval_lev_vocab` + `agg_lev_vocab` + `make_lev_vocab_fig` (superseded by
`src/eval_lev_scaling.py`), `make_item_analysis`, `make_item_plot`, `plot_item_confusion_cues`.
book figures: `make_figures`, `make_waterfall`, `make_ladder_figs`, `make_scaling_figs`,
`make_arch_fig`, `make_confusion`, `make_dev_figure`, `make_encoder_ladder_fig`,
`make_disagree_frames`, `make_gemini_examples`, `plot_gemini_val`, `data_stats`, `lex_oracle_fig`.

**Cue analyses (book chapter 5 — excluded from the paper by decision)**
`build_cue_manifest`, `build_combined_cues`, `build_speaker_manifests`, `build_utterance_cues`,
`build_pose_face`, `build_pose_filter_manifests`, `build_pose_gaze`, `build_pose_gesture`,
`build_pose_targets`, `pose_align_dur`, `pose_boxscore`, `pose_compare`, `pose_coverage`,
`pose_cue_audit`, `pose_cues_full`, `pose_diag`, `pose_fp_check`, `pose_hand_camera`,
`pose_hands_split`, `pose_infer`, `pose_pointing`, `pose_yolo_test`, `sanity_pose`,
`cue_audit`, `cue_gemini_corr`, `speaker_for_pairs`, `discourse_for_pairs`,
`frame_center_for_pairs`, `prosody_for_pairs`, `add_prosody`, `imu_camera_pitch`,
`gen_point_overlays`, `characterize_persons`, `caregiver_close_for_pairs`.
(`src/pose_lib.py` stays: `blur_faces.py` imports it for the fig1 frames.)

**Superseded by the paper's final versions**
`build_pairs` → `src/build_pairs_2026.py`; `filter_english`, `langid_utterances`,
`langid_validate`, `build_english_controls` (fastText / transcript-era English filtering) →
the audio-based filter in bv-annotations (`build_english_filter_audio.py`).

Runners for these eras are in `runners/archive/`.
