# Handoff: human check of the Gemini referential-alignment annotations

> **Status 2026-09-09:** built. Scripts, app, and operating instructions are in `human_check/`
> (see `human_check/README.md`). Sample + frames are staged on ccn2-14; the Cloud Run + IAP
> deployment (shareable without cluster access) needs Mike's `gcloud auth login` once.

Goal: a small, lab-shareable app in which people rate ~1,000 (frame, utterance) pairs so we can
report how good the Gemini alignment annotation is. This annotation is the paper's key mechanism
(the aligned ~10% of pairs carries essentially all the word learning; removing them collapses
accuracy from 81.6 to 42.7 for L-OTS), so a reviewer will ask how much to trust it. Deliverables:
(1) a sampling script, (2) a frame-pull script, (3) the app, (4) an analysis script producing the
SI numbers. Written 2026-09-09 by the pipeline session; ask Mike for anything marked **[Mike]**.

## 1. What Gemini annotated

- Model: `gemini-2.5-flash` on Vertex AI (project `hs-hs-langcog-gemini`; Vertex terms = no
  training on inputs, zero retention — the IRB-relevant property). Code and prompt:
  `~/Projects/bv-annotations/alignment/gemini_align.py` (PROMPT at line 57; `load_jpeg` downsizes
  the frame's longest edge to `max_px` before sending — check the value used for the full run so
  the app can optionally show raters the same resolution).
- Input per item: ONE 1-fps frame (the utterance's temporal midpoint) + the utterance text.
- Prompt (verbatim):
  > Return JSON with:
  > - "alignment": integer 0-100 for how strongly the utterance refers to a concrete object
  >   VISIBLE in the frame. 100 = clearly names a prominent, plainly-visible, central object;
  >   ~50 = the named object is present but small, partial, or one of many; 0 = no visible
  >   referent (small talk, the object is absent or unidentifiable, or the utterance is not
  >   about a concrete object). Use the full range, not just the ends.
  > - "referent": the single visible object referred to, as a lowercase common noun
  >   (e.g. "cup", "dog"); "" if alignment is low or none applies.
  > Utterance: <text>
- Output table: `/ccn2b/dataset/babyview/2026.1/outputs/annotations/referent/gemini_2026.1.parquet`
  columns `video_id, frame_idx, text, child_id, alignment, referent, error`; 1,838,134 rows
  (1,838,061 scored). Distribution: 90% are exactly 0; the rest pile up at 50–100
  (80: 47,770 · 100: 40,480 · 50: 36,139 · 70: 30,933 · 90: 27,115); referent non-empty for
  185,449 (10.1%). The paper's rule: **aligned = alignment ≥ 50** (171,782 pairs in the final
  corpus; the referent is non-empty for essentially all of those).
- The FINAL corpus (what the models train on) is `manifests/bv26_pairs_en_audio.parquet`
  (1,686,105 pairs, 48 children; audio-based <50% English video filter applied). Sample from the
  intersection of this manifest with the scored table, not from the raw scored table.
- Join key is the triple `(video_id, frame_idx, text)`. It repeats for ~31k rows (same text twice
  in the same second) — dedupe before sampling, and never join on text alone.

## 2. Where the frames are (and the privacy rule)

- Canonical: `/ccn2b/dataset/babyview/2026.1/extracted_frames_1fps/<video_id>/<frame_idx:05d>.jpg`
  (`frame_path()` in `vlm-headcam/src/common.py` resolves this; set
  `BABYVIEW_ROOT=/ccn2b/dataset/babyview/2026.1`). Fast local copy on the node:
  `/data2/mcfrank/frames_1fps_local/<video_id>/<frame_idx:05d>.jpg` (16,306 videos).
- **These are human-subjects headcam frames of children and families. They must not leave the
  cluster / Stanford-controlled storage: no Google Forms, no Qualtrics image uploads, no GitHub,
  no HF, no claude.ai artifacts, no laptop copies handed around.** The app therefore runs ON
  ccn2-14 (or another lab machine with the data) and lab mates reach it through an ssh tunnel
  (`ssh -L 8501:localhost:8501 ccn2-14`, then http://localhost:8501). **[Mike]** confirm whether a
  lab-internal web host counts as "on the cluster" for this purpose; if not, ssh tunnel it is.
- The node env with pandas/PIL/torch: `/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python`. Check
  whether `streamlit` is installed there; if not, a Flask/FastAPI app with one HTML page is fine
  (pip install into a venv under `/data2/mcfrank/`, not into the shared env).

## 3. Sampling design (1,000 items) — recommendation

The question the paper needs answered is "how accurate is the ≥50 decision, and is the referent
right?", plus calibration across the range. Stratify by Gemini score so every bin has enough
items, and keep the sampling weights so corpus-level precision/recall can be reweighted:

| stratum (Gemini alignment) | n | why |
|---|---|---|
| 0, utterance contains a concrete noun (CDI/Konkle object word) | 200 | the hard negatives: false-negative rate where a referent was plausible |
| 0, other | 100 | ordinary negatives (small talk) |
| 50 | 150 | the threshold bin — most consequential |
| 70 | 150 | |
| 80 | 150 | |
| 90 | 100 | |
| 100 | 150 | |

Constraints: at most ~30 items per child (48 children), at most one item per video-minute,
dedupe near-identical frames (the eval builder used an embedding cosine > 0.92 rule; simpler:
one item per video per stratum), drop utterances with fewer than 2 tokens or > 20 tokens, drop
the `error` rows. Fixed random seed; write `sample.parquet` with the stratum, the sampling
weight, and Gemini's score/referent. Randomize presentation order per rater. A "concrete noun"
list is easy to derive from `manifests/eval_frames_konkle.parquet` categories + the CDI object
words in `src/make_cdi_categories.py`.

## 4. Pull script

Reads `sample.parquet`, copies the 1,000 JPEGs into `app/frames/<item_id>.jpg` on the node
(rename to opaque ids so the app never exposes video ids / child ids to raters), and writes
`app/items.json` (item_id, utterance, Gemini score + referent kept SERVER-SIDE only). Optionally
also store the ±1 s neighbours for a "context" toggle — but see §5: for the blind task the rater
should see exactly what Gemini saw (one frame + text).

## 5. The choice point: response format

Two designs; they answer different questions.

**A. "Is Gemini right?"** — rater sees frame + utterance + Gemini's score and referent, answers
agree / disagree (+ corrected referent). Fast (~5 s/item), but anchored: it measures perceived
plausibility of Gemini's answer, and agreement will be inflated. Can't estimate false negatives
independently, can't give calibration.

**B. "Same task as Gemini"** (blind) — rater sees ONLY frame + utterance and answers the same two
questions Gemini did. Slower (~12 s/item) but gives an independent human judgment: agreement /
κ at the ≥50 threshold, precision and recall of Gemini against the human majority, calibration
(human mean by Gemini bin), and referent match rate. This is what a reviewer wants.

**Recommendation: B, with a coarse response matching the prompt's anchors** rather than a 0–100
slider (humans don't use the slider reliably):
- Q1 "Does the utterance refer to a concrete object visible in the frame?" — three buttons:
  **No visible referent** (→0) · **Present but small / partial / one of many** (→50) ·
  **Clearly present** (→100); plus a **Can't tell** flag (unusable frame, blur, occlusion).
- Q2 "What object?" — free-text noun, shown only if Q1 ≠ No; store lowercased; match to Gemini's
  referent by exact / singularised / WordNet-synonym match at analysis time.
- 2–3 raters per item (κ needs ≥2; 3 gives a majority). 1,000 items × 3 raters × 12 s ≈ 10 h of
  lab time total; make the app resumable so people do 100 at a sitting.
- Optional pass A afterwards on the disagreements only (cheap, and informative about WHY).

Per-rater login by name (no auth needed behind the tunnel), progress bar, keyboard shortcuts
(1/2/3, Enter), randomised order per rater, responses appended to a JSONL/SQLite on the node
with timestamps, never written into the frames dir.

## 6. Analysis script (→ SI numbers)

From responses + `sample.parquet`:
- Inter-rater agreement (κ on the 3-level scale and on the binary ≥50 mapping).
- Gemini vs human majority: Spearman on 0/50/100 vs Gemini score; κ on binary; **precision and
  recall of Gemini's ≥50 rule** with sampling weights → corpus-level estimates.
- Calibration table/plot: human mean score per Gemini bin (0-noun, 0-other, 50, 70, 80, 90, 100).
- Referent accuracy among items both call aligned (exact / lemma / synonym).
- Per-child spread (any child with systematically worse agreement?).
Write `results/gemini_human_check.csv` (aggregate numbers only — no frames, no ids) into
vlm-headcam so `make_methods_numbers.py` can pick the headline numbers up as macros.

## 7. Things not to do

- Don't put frames anywhere outside the cluster (see §2). Don't commit frames, `items.json`
  with video ids, or response files with rater names to the vlm-headcam repo; commit the scripts
  and the aggregate CSV only.
- Don't sample from the raw 1.84M scored table (includes videos the language filter removed);
  use the final corpus intersection.
- Don't show raters Gemini's answer in the blind pass; if you add a "context" toggle (±1 s
  frames), log its use — it gives the human more information than Gemini had.
