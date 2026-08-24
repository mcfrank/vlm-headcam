# BabyView 2026.1 — outputs/ index

16,306 videos, 51 children, 2,701 hours (ages ~3–54 months). Consolidated 2026-08-24;
verified by `verify_release.py` (mcfrank/vlm-headcam). Questions: Mike Frank / #babyview.

## Join keys — read before joining anything
- Universal video id: the release name, e.g. `S00220001_2024-02-05_1_recfcw2yqs02gdskT`
  (may end `_rotated`). It names frames dirs, transcript rows, pose dirs, and all annotations.
- `release_index.tsv` maps video_id ↔ Airtable rec_id ↔ subject_id (+ rotated flag).
  Use it; do not parse ids by hand.
- Within a video: `frame_idx` = second (1 fps); `utterance_id` from the parsed transcript.
- NEVER join layers on utterance text — transcript pipelines segment differently and a text
  join silently drops ~half the rows.
- Airtable `percent_english` is a PROPORTION (0–1), not a percent.

## Layers
| file / dir | what | key | coverage |
|---|---|---|---|
| ../extracted_frames_1fps/ | 1 fps jpgs | video_id/frame_idx | 16,306/16,306 |
| merged_transcripts_parsed.csv | ASR utterances + spaCy tokens | video_id, utterance_id | 16,008 videos have speech |
| pose_1fps_bbox_limbs.parquet | whole-body pose flat table | video_id, frame_idx, person_idx | 16,296 (10 sub-second clips have no frames) |
| pose_1fps/ + README_pose_1fps.md | per-frame 133-kpt pkls | dir = video_id | same |
| image_embeddings/dinov3b_grid4x4/ | DINOv3-B 4×4 region grid | index.parquet row ↔ (video_id, frame_idx) | all 1,745,489 pair-frames |
| annotations/referent/ | Gemini alignment + referent word | see its README | 99.996% of pairs |
| annotations/language/ | per-utterance language ID | see its README | 100% of utterances |
| videos_airtable_2026-07-24.csv | registry snapshot | unique_video_id (rec id) | — |
| MANIFEST.tsv / migration_log.tsv | checksums / provenance | — | — |

## Rules
- Run `verify_release.py` before building on this tree.
- Don't add files without updating MANIFEST.tsv.
- Frames show children's faces: no frame, clip, or audio leaves the cluster.
