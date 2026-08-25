"""The Fig-1 data funnel, machine-readable: every corpus-defining step with its rule and
count, generated from the release artifacts themselves (never hand-typed). Commit the JSON;
the figures session renders it. Rerun after any filter change.

usage: python src/make_pipeline_counts.py  -> results/pipeline_counts.json"""
import json
from pathlib import Path

import pandas as pd

O = Path("/ccn2b/dataset/babyview/2026.1/outputs")
steps = []

idx = pd.read_csv(O / "release_index.tsv", sep="\t")
steps.append(dict(step="release", rule="BabyView 2026.1 release (Airtable-tagged)",
                  videos=len(idx), children=int(idx.subject_id.nunique())))
t = pd.read_cssv if False else pd.read_csv(O / "merged_transcripts_parsed.csv",
                                           usecols=["video_id", "utterance_id"]).drop_duplicates()
steps.append(dict(step="transcribed", rule="ASR utterances (videos with speech)",
                  videos=int(t.video_id.nunique()), utterances=len(t)))
P = pd.read_parquet(O / "annotations/referent/pairs_2026.1.parquet")
steps.append(dict(step="pairs", rule="utterance x frame-at-utterance-time; frame on disk",
                  pairs=len(P), frames=int(P.groupby(["video_id", "frame_idx"]).ngroups)))
G = pd.read_parquet(O / "annotations/referent/gemini_2026.1.parquet")
steps.append(dict(step="referent-scored", rule="Gemini alignment 0-100 + referent word",
                  pairs=int(G.alignment.notna().sum())))
dec = pd.read_csv(O / "annotations/language/audio/video_decisions_audio_2026.1.csv")
en = pd.read_parquet("/data2/mcfrank/vlm-headcam/manifests/bv26_pairs_en_audio.parquet")
steps.append(dict(step="english-filter",
                  rule="drop videos <50% audio-measured English (Gemini audio ID); "
                       "drop transcript-flagged non-English utterances",
                  videos_dropped=int((dec.decision == "drop-non-english").sum()),
                  pairs=len(en),
                  children=int(en.video_id.str.split("_").str[0].nunique())))
steps.append(dict(step="training-corpus", rule="final word-learning corpus",
                  pairs=len(en)))

# ---- panel-B referential funnel (WITHIN the filtered training corpus, so percentages are
# internally consistent): all pairs -> Gemini says a referent is visible -> referent word
# actually spoken in the utterance (same tokenize+singularize test as the t15 manifests).
import re as _re
import sys as _sys
_sys.path.insert(0, "src")
from common import tokenize

def _sing(w):
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if _re.search(r"(ses|xes|zes|ches|shes)$", w): return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w

GF = G.merge(en[["video_id", "frame_idx", "text"]].drop_duplicates(),
             on=["video_id", "frame_idx", "text"], how="inner")
al = GF[GF.alignment >= 50]
steps.append(dict(step="aligned", rule="Gemini alignment >= 50 (a named object is visible)",
                  pairs=len(al), of_corpus=round(len(al) / len(en), 4)))
def _spoken(row):
    r = _sing(str(row.referent).strip().lower())
    toks = set(tokenize(row.text)); toks |= {_sing(t) for t in toks}
    return bool(r) and r in toks
ref = al[al.referent.fillna("").str.len() > 0]
sp = ref[[_spoken(r) for r in ref.itertuples()]]
steps.append(dict(step="referent-spoken", rule="referent word appears in the utterance "
                  "(tokenized + singularized match)", pairs=len(sp),
                  of_corpus=round(len(sp) / len(en), 4)))
out = dict(release="2026.1", generated_from="release artifacts (see MANIFEST.tsv)",
           note="regenerate with src/make_pipeline_counts.py after any filter change",
           steps=steps)
Path("results").mkdir(exist_ok=True)
Path("results/pipeline_counts.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2)[:900])
