"""Release diagnostics: one small, face-free aggregates bundle describing every layer of a
BabyView release (registry, frames, transcripts, pose, referent annotation, language annotation,
embeddings), so the supplement can render off-cluster.

Runs ON ccn2 (the data are restricted and large); writes ~1 MB of parquet that IS committed.
Every layer is optional — a layer that is missing or still running is reported as absent rather
than crashing, and its coverage shows up as a gap in the coverage table.

usage: python src/make_diagnostics.py --release 2026.1 --out diagnostics/2026.1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--release", default="2026.1")
ap.add_argument("--root", default="/ccn2b/dataset/babyview/2026.1")
ap.add_argument("--airtable", default="/ccn2b/dataset/babyview/2026.1/outputs/videos_airtable_2026-07-24.csv")
ap.add_argument("--transcript", default="/ccn2b/dataset/babyview/2026.1/outputs/merged_transcripts_parsed.csv")
ap.add_argument("--pose", default="/ccn2b/dataset/babyview/2026.1/outputs/pose_1fps_bbox_limbs.parquet")
ap.add_argument("--referent", default="/ccn2b/dataset/babyview/2026.1/outputs/annotations/referent/gemini_2026.1.parquet")
ap.add_argument("--language", default="/ccn2b/dataset/babyview/2026.1/outputs/annotations/language/lang_2026.1.parquet")
ap.add_argument("--pairs", default="/ccn2b/dataset/babyview/2026.1/outputs/annotations/referent/pairs_2026.1.parquet")
ap.add_argument("--embed-glob", default="/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/dinov3b_grid4x4/index.parquet")
ap.add_argument("--out", default="diagnostics/2026.1")
a = ap.parse_args()

OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
prov = {"release": a.release, "sources": {}, "counts": {}}


def note(layer, path, n, ok=True):
    prov["sources"][layer] = {"path": str(path), "present": bool(ok)}
    prov["counts"][layer] = int(n)
    print(f"  {layer:12s} {'ok ' if ok else 'MISSING'} n={n:,}" if ok else f"  {layer:12s} MISSING ({path})")


# ---- 1. registry: the spine. KEYED BY THE RELEASE NAME (the long composite id every layer
# uses) via release_index.tsv; Airtable metadata joins on the bare rec-id. An earlier version
# used unique_video_id (bare rec-id) as the spine, so every layer merge silently matched
# nothing and fillna(0) hid it — the same key-mismatch class as the text-join bug.
print("registry")
ridx = pd.read_csv(Path(a.root) / "outputs/release_index.tsv", sep="\t")
at = pd.read_csv(a.airtable, low_memory=False)
at = at[at.release.astype(str).str.contains(a.release, na=False)].copy()
at["rec_id"] = at.unique_video_id.astype(str)
at["age_years"] = pd.to_numeric(at["age (years)"], errors="coerce")
at["hours"] = pd.to_numeric(at.duration_hours, errors="coerce")
# Airtable percent_english is a PROPORTION (0-1) despite the name; convert to percent here so
# no downstream consumer plots 1.0 as "1%" (which crushed all 51 children to x~0 once already)
at["survey_pct_english"] = pd.to_numeric(at.percent_english, errors="coerce") * 100
V = ridx.merge(at[["rec_id", "age_years", "hours", "camera", "survey_pct_english",
                   "participant_languages", "date"]].drop_duplicates("rec_id"),
               on="rec_id", how="left")
V["child"] = V.subject_id.astype(str)
V["age_months"] = V.age_years * 12
assert V.hours.notna().mean() > 0.99, "registry join failed — rec_id mismatch"
note("registry", a.airtable, len(V))
# ---- 2. transcripts: utterances + tokens per video ----------------------------------------
print("transcripts")
try:
    tr = pd.read_csv(a.transcript, usecols=["video_id", "utterance_id", "utterance", "speaker"],
                     low_memory=False)
    utt = tr.drop_duplicates(["video_id", "utterance_id"])
    g = utt.groupby("video_id").agg(n_utterances=("utterance_id", "size"))
    g["n_words"] = utt.assign(w=utt.utterance.astype(str).str.split().str.len()).groupby("video_id").w.sum()
    V = V.merge(g, on="video_id", how="left")
    note("transcripts", a.transcript, len(utt))
    # utterance-length distribution (pre-binned; the raw text never leaves the cluster)
    nw = utt.utterance.astype(str).str.split().str.len().clip(0, 30)
    pd.DataFrame({"n_words": nw.value_counts().sort_index().index,
                  "count": nw.value_counts().sort_index().values}).to_parquet(OUT / "dist_utterance_len.parquet", index=False)
except Exception as e:
    note("transcripts", a.transcript, 0, ok=False); print(f"    {e}")

# ---- 3. pose: person detections per video (rekeyed parquet; person_detected==1 only) -------
print("pose")
try:
    pp = pd.read_parquet(a.pose, columns=["video_id", "frame_idx", "person_detected"])
    det = pp[pp.person_detected == 1]
    P = det.groupby("video_id").agg(pose_persons=("person_detected", "size"),
                                    pose_frames_with_person=("frame_idx", "nunique")).reset_index()
    P = P.merge(pp.groupby("video_id").frame_idx.nunique().rename("pose_frames").reset_index(),
                on="video_id", how="right").fillna(0)
    V = V.merge(P, on="video_id", how="left")
    note("pose", a.pose, int(P.pose_persons.sum()))
    npf = det.groupby(["video_id", "frame_idx"]).size().clip(0, 10).value_counts().sort_index()
    pd.DataFrame({"persons_in_frame": npf.index, "count": npf.values}).to_parquet(
        OUT / "dist_persons_per_frame.parquet", index=False)
    # split: a detection with a visible face or body is a social partner; hands-only detections
    # are largely the wearing child's own hands entering the frame (20% of all detections)
    soc = pd.read_parquet(a.pose, columns=["video_id", "frame_idx", "person_detected",
                                           "face_in_image", "body_in_image"])
    soc = soc[(soc.person_detected == 1) &
              (soc.face_in_image.astype(bool) | soc.body_in_image.astype(bool))]
    nsf = soc.groupby(["video_id", "frame_idx"]).size().clip(0, 10).value_counts().sort_index()
    pd.DataFrame({"persons_in_frame": nsf.index, "count": nsf.values}).to_parquet(
        OUT / "dist_partners_per_frame.parquet", index=False)
    P2 = soc.groupby("video_id").frame_idx.nunique().rename("pose_frames_with_partner").reset_index()
    V = V.merge(P2, on="video_id", how="left")
except Exception as e:
    note("pose", a.pose, 0, ok=False); print(f"    {e}")

# ---- 4. referent annotation (Gemini alignment + referent word) -----------------------------
print("referent")
try:
    r = pd.read_parquet(a.referent)
    ok = r[r.alignment.notna()]
    g = ok.groupby("video_id").agg(ref_scored=("alignment", "size"),
                                   ref_mean_alignment=("alignment", "mean"))
    g["ref_has_referent"] = ok.assign(h=ok.referent.astype(str).str.len() > 0).groupby("video_id").h.sum()
    V = V.merge(g, on="video_id", how="left")
    note("referent", a.referent, len(ok))
    al = ok.alignment.clip(0, 100).round(-1).value_counts().sort_index()
    pd.DataFrame({"alignment_bin": al.index, "count": al.values}).to_parquet(OUT / "dist_alignment.parquet", index=False)
except Exception as e:
    note("referent", a.referent, 0, ok=False); print(f"    {e}")

# ---- 5. language annotation ----------------------------------------------------------------
print("language")
try:
    L = pd.read_parquet(a.language)
    L["en"] = L.is_english.fillna(False)
    g = L.groupby("video_id").agg(lang_utterances=("en", "size"), lang_english=("en", "sum"))
    g["lang_agree"] = L.groupby("video_id").agree.mean() if "agree" in L else np.nan
    V = V.merge(g, on="video_id", how="left")
    note("language", a.language, len(L))
except Exception as e:
    note("language", a.language, 0, ok=False); print(f"    {e}")

# ---- 6. embeddings + training pairs (coverage only) -----------------------------------------
print("embeddings / pairs")
import glob
try:
    idx = pd.concat([pd.read_parquet(f, columns=["video_id"]) for f in glob.glob(a.embed_glob)])
    e = idx.groupby("video_id").size().rename("emb_frames").reset_index()
    V = V.merge(e, on="video_id", how="left")
    note("embeddings", a.embed_glob, len(idx))
except Exception as ex:
    note("embeddings", a.embed_glob, 0, ok=False); print(f"    {ex}")
try:
    pr = pd.read_parquet(a.pairs, columns=["video_id"])
    V = V.merge(pr.groupby("video_id").size().rename("n_pairs").reset_index(), on="video_id", how="left")
    note("pairs", a.pairs, len(pr))
except Exception as ex:
    note("pairs", a.pairs, 0, ok=False); print(f"    {ex}")

# ---- 7. roll up + write ----------------------------------------------------------------------
if "n_utterances" in V and V.n_utterances.fillna(0).sum() == 0:
    raise SystemExit("ABORT: transcript merge matched nothing — spine key mismatch")
NUM = [c for c in V.columns if V[c].dtype.kind in "if" and c not in ("age_years", "age_months", "survey_pct_english")]
V[NUM] = V[NUM].fillna(0)
V.to_parquet(OUT / "video_level.parquet", index=False)

agg = {"n_videos": ("video_id", "size"), "hours": ("hours", "sum"),
       "age_min": ("age_months", "min"), "age_max": ("age_months", "max"),
       "age_median": ("age_months", "median"), "survey_pct_english": ("survey_pct_english", "median")}
for c in ["n_utterances", "n_words", "pose_persons", "pose_frames_with_person",
          "ref_scored", "ref_has_referent", "lang_utterances", "lang_english", "emb_frames", "n_pairs"]:
    if c in V: agg[c] = (c, "sum")
C = V.groupby("child").agg(**agg).reset_index()
if {"lang_english", "lang_utterances"} <= set(C.columns):
    C["measured_pct_english"] = 100 * C.lang_english / C.lang_utterances.replace(0, np.nan)
if {"ref_mean_alignment"} <= set(V.columns):
    C = C.merge(V.groupby("child").ref_mean_alignment.mean().reset_index(), on="child", how="left")
C.to_parquet(OUT / "child_level.parquet", index=False)

prov["n_children"] = int(V.child.nunique())
prov["n_videos"] = int(len(V))
(OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
print(f"\nwrote {OUT}/  ({len(V):,} videos, {V.child.nunique()} children)")
print(f"  total size: {sum(f.stat().st_size for f in OUT.glob('*')) / 1e6:.2f} MB")
