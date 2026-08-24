# annotations/referent/ — Gemini referent-alignment annotation

Per **utterance–frame training pair**: does the utterance refer to something visible in the
frame, and to what?

## Files
- `gemini_2026.1.parquet` — 1,838,134 rows; 1,838,061 scored (99.996%).
  Columns: video_id, frame_idx, text, alignment (0–100), referent (word or ""), child_id, …
- `pairs_2026.1.parquet` — the pair manifest the annotation covers (1,838,134 rows).
  Keys: video_id, **utterance_id**, frame_idx; built by build_pairs_2026.py (vlm-headcam)
  from merged_transcripts_parsed.csv; 154 pairs whose frame is missing on disk are excluded.
- `frames_2026.1.parquet` — the 1,745,489 unique frames of those pairs.

## Method + provenance
Gemini-2.5-Flash on **Vertex AI** (project hs-hs-langcog-gemini — no Google training on our
data), 2026-08-23/24, prompt + runner: vlm-headcam src/gemini_align.py. Alignment 0 = nothing
named is visible; ≥50 (9.2% of pairs) = referential; the `referent` field names the object.
Validation against CLIP and humans: book ch. 2/3.

## Join
To transcripts/language: (video_id, utterance_id) via pairs_2026.1.parquet.
To frames/embeddings: (video_id, frame_idx). Do NOT join on text.
