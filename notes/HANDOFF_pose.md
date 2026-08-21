# Pose detection — pointer

The whole-body pose pipeline (YOLO12x → mmpose RTMW-x, 133-kpt COCO-WholeBody) and its runbooks live
in the **bv-annotations** repo (`~/Projects/bv-annotations`, node `~/bv-annotations`), not here:

- `docs/handoff-pose-2026.1.md` — **start here**: state as of 2026-08-21, what 2025.2 produced, why
  RTMW-x, the 2026.1 plan (env is gone → rebuild; frames not yet extracted; chunked runs).
- `pose/README.md`, `pose/SETUP.md`, `pose/qa/README.md`, `docs/adding-a-release.md`.

2025.2 outputs this repo consumes: `/ccn2/dataset/babyview/2025.2/outputs/pose_1fps/` (pkls) and
`…/pose_1fps_bbox_limbs.csv` (6.5M rows). Readers/consumers here: `src/pose_lib.py`,
`src/build_pose_{targets,face,gaze,gesture}.py`, `src/pose_cue_audit.py` (the cue nulls of ch5).
The untracked `src/pose_{diag,fp_check,boxscore,yolo_test}.py` are the calibration throwaways that
justified the model choice.
