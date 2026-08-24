# BabyView 2026.1 — data layout and layer inventory

Survey date 2026-08-23. 2026.1 is the consolidation release: **16,306 videos, 51 children,
2,701 hours, ages 0.24–4.51 y** (median 1.42 y). 8,566 videos are the 2025.2 set (identical
files); 11,735 are new. This file is the map; each layer should also carry its own README.

## The join key (read this first)

Every layer resolves to the Airtable **rec-id** (`unique_video_id`, e.g. `recuYc5uzv9dxPSYt`),
and from there to a child via `subject_id`. Within a video:

| grain | key |
|---|---|
| utterance | `(video_id, utterance_id)` |
| frame | `(video_id, frame_idx)` — frame index **= second** (1 fps) |

**Do not join on utterance text.** The CLIP-era pair pipeline (`full_clip_results.csv`) and the
parsed-transcript pipeline (`merged_transcripts_parsed.csv`) segment utterances differently: on
2025.2 only 47% of texts match, and a naive text join silently drops the other 53% while looking
like a content decision. This cost us a bogus "flat alignment filter" result once already.

**The pose layer is the exception** and needs a crosswalk: it keys on
`superseded_gcp_name_feb25` (`S00320003_2024-11-03_3_recuYc5uzv9dxPSYt`, sometimes
`..._rotated`) plus a `H:MM:SS` timestamp. Recover the standard keys with
`video_id = re.search(r"rec[A-Za-z0-9]{14,}", name)` and `frame_idx = seconds(timestamp)`.
Verified 100% of recovered ids are in the release. Anchoring the regex to end-of-string drops
the 7.8% `_rotated` rows.

## Where each layer lives

2026.1 is currently split across four roots. Consolidating these is the open work.

| Layer | Location | Key | Status |
|---|---|---|---|
| frames (1 fps) | `/ccn2b/…/2026.1/extracted_frames_1fps/` | `video_id/frame_idx` | canonical |
| 10 s clips, codes | `/ccn2b/…/2026.1/{10s_clips,codes}/` | — | canonical |
| pose (133-kpt) | `/ccn2b/…/2026.1/outputs/pose_1fps/` + `pose_1fps_bbox_limbs.csv` | gcp-name + time | canonical, **README exemplar** |
| registry (Airtable) | `/ccn2b/…/2026.1/outputs/videos_airtable_2026-07-24.csv` | `unique_video_id` | canonical |
| release membership | `/ccn2b/…/2026.1/outputs/release_2026_1_ids.txt` (16,306) + `excluded_…txt` (3,995) | — | canonical |
| DINOv3-B embeddings | `/ccn2b/…/2026.1/outputs/image_embeddings/dinov3b_grid4x4/` | `(video_id, frame_idx)` | ours |
| audio | `/ccn2a/…/2026.1/mp3/` | `video_id` | **all but absent — 1 child** |
| transcripts (parsed) | `/ccn2a/…/2026.1/outputs/merged_transcripts_parsed.csv` | `(video_id, utterance_id)` | **split off** |
| language annotation | `/ccn2/dataset/babyview/annotations/language/lang_2026.1.parquet` | `(video_id, utterance_id)` | **outside the release tree** |
| referent annotation | `/data2/mcfrank/vlm-headcam/scored/bv2026_gemini.parquet` | `(video_id, frame_idx, text)` | **node-local scratch — at risk** |

### The risk worth acting on
The **referent annotation** (Gemini alignment + referent word over 1.84M utterance–frame pairs,
~$255 of credits, the most expensive derived asset in the project) sits on **node14's local
NVMe**: not shared, not backed up, not discoverable. It should move under the release before
anything else. The language annotation has the same problem in milder form — on shared storage,
but in a tree nobody browsing 2026.1 would look in.

## Target layout

```
/ccn2b/dataset/babyview/2026.1/
  extracted_frames_1fps/
  10s_clips/  codes/  transcripts/
  outputs/
    merged_transcripts_parsed.csv        <- relocate from /ccn2a (or symlink)
    pose_1fps/  pose_1fps_bbox_limbs.csv  README_pose_1fps.md
    image_embeddings/dinov3b_grid4x4/     README.md
    annotations/
      referent/  gemini_2026.1.parquet    README.md   <- off node-local scratch
      language/  lang_2026.1.parquet      README.md
    videos_airtable_2026-07-24.csv
    release_2026_1_ids.txt  excluded_from_2026_1_ids.txt
    README.md                             <- index of every layer
```

Each layer README states: what it is, how it was produced (model, env pins, date), **its join
key**, coverage against the 16,306, known gaps, and the reader/consumer code. `README_pose_1fps.md`
is the standard to match — it documents membership, the quarantined non-release videos, env pins,
and even the 12 sub-second clips that yield no frames.

## Sizes (2026-08-23)

| | |
|---|---|
| `extracted_frames_1fps/` | 609 G |
| `outputs/pose_1fps/` | 24 G |
| `outputs/pose_1fps_bbox_limbs.csv` | 2.6 G |
| `/ccn2a/…/2026.1/mp3/` | **671 K (1 child)** — cf. 2025.2's 121 G / 37 children |

## Known gaps / caveats

- **Audio is effectively missing from 2026.1.** `/ccn2a/…/2026.1/mp3/` holds one child
  (S00220001); 2025.2 has 37 children / 121 G. The *transcripts* are complete (all 1,838,288
  utterances in `merged_transcripts_parsed.csv`), so ASR ran upstream and only its output was
  copied over. Consequence: **no audio-derived feature can be recomputed on 2026.1** without
  re-pulling audio from GCS — including the ch5 prosody cue (per-word RMS energy). Fine for
  everything we currently plan (vision + text), but a blocker for any prosody follow-up.
- 12 sub-second clips (<0.5 s; 11× S00270001, 1× S00560001) yield no 1 fps frame and have no
  directory. Listed as `fail_rc0` in `frames_manifest*.tsv`.
- 3,995 processed-but-not-in-release videos live in `/ccn2b/dataset/babyview/_mcfrank_not_in_2026.1/`.
  Do not mix them in.
- 154 utterance–frame pairs reference frames absent from disk; `build_pairs_2026.py --require-frame`
  drops them (1,838,288 → 1,838,134).
