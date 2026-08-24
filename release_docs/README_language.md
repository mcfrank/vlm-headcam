# annotations/language/ — utterance-level language identification

Per **utterance**: the language spoken, for filtering non-English input and for language-
environment description. A standalone resource for any BabyView user.

## Files (per release R = 2025.2, 2026.1)
- `lang_R.parquet` — the one to use. Keys: video_id, utterance_id. Columns: text, nwords,
  lang_p0/lang_p1 (two independent passes), agree, lang, is_english, mixed, undecidable.
  `lang` is NULL where the passes disagree (0.6%).
- `lang_R_p{0,1}.{jsonl,parquet}` — raw per-pass outputs (resumable checkpoints).
- `video_decisions_R.csv` — per-video English fraction + filter decision.
- `usage_R_p*.json` — token/cost accounting.

## Method + reliability (2026.1)
Two independent Gemini-2.5-Flash passes on Vertex AI (no Google training), chunked with
video context, offset chunk boundaries between passes (bv-annotations/language/).
1,838,288/1,838,288 utterances; **99.4% pass agreement; 99.05% English among agreed;
0.12% mixed; 0.79% undecidable.** Known biases + fastText comparison (99% precise on
English, 58% on non-English here): bv-annotations/language/README.md.

## Join
(video_id, utterance_id) to the parsed transcript and to the referent pairs manifest.
NEVER on text. Survey comparison: registry `percent_english` is a 0–1 proportion.
