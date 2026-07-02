"""Ceiling test: is the region-MIL oracle capped by the weak TEXT signal (bag-of-words real
utterances) rather than by alignment? Build the aligned oracle set two ways on the SAME
pairs: text = full utterance vs text = NOUNS only. Nouns are identified by a corpus-wide
lexicon (surface forms predominantly tagged NOUN/PROPN in the parsed transcripts) rather than
a per-utterance id join (the two pipelines segment utterances differently, so utterance_id
does not align). If noun-only lifts 4AFC toward the clean-label topline (~72), the ceiling
is text-limited, not alignment-limited."""
import re
import numpy as np, pandas as pd
from pathlib import Path

BV = Path("/ccn2a/dataset/babyview/2025.2")
W = Path("/data2/mcfrank/vlm-headcam")
HELD = "S00360001"
_tok = re.compile(r"[a-z]+")

# --- noun lexicon: surface forms that are predominantly nouns across the corpus ---
mt = pd.read_csv(BV / "outputs/merged_transcripts_parsed.csv",
                 usecols=["spacy_token_text", "spacy_pos"], low_memory=False)
mt["w"] = mt.spacy_token_text.astype(str).str.lower()
mt = mt[mt.w.str.fullmatch(r"[a-z]+")]
tot = mt.groupby("w").size()
nn = mt[mt.spacy_pos.isin(["NOUN", "PROPN"])].groupby("w").size()
pnoun = (nn / tot).fillna(0.0)
noun_lex = set(pnoun[(pnoun >= 0.5) & (tot >= 3)].index)
print(f"noun lexicon: {len(noun_lex)} surface forms (e.g. {sorted(list(noun_lex))[:8]})")
for w in ["can", "book", "present", "kitchen", "go", "the", "cat"]:
    print(f"  P(noun|{w})={pnoun.get(w, 0):.2f}  in_lex={w in noun_lex}")

def noun_text(t):
    return " ".join(w for w in _tok.findall(str(t).lower()) if w in noun_lex)

# --- aligned oracle set ---
cr = pd.read_csv(BV / "outputs/full_clip_results.csv", usecols=[
    "child_id", "video_name", "utterance", "utterance_start_time",
    "utterance_end_time", "clip_score_max"]).rename(
    columns={"video_name": "video_id", "utterance": "text"})
cr = cr.dropna(subset=["clip_score_max", "utterance_start_time", "utterance_end_time"])
cr = cr[(cr.clip_score_max > 0.24) & (cr.child_id.astype(str) != HELD)]
cr["frame_idx"] = ((cr.utterance_start_time + cr.utterance_end_time) / 2).astype(int)
cr["child_id"] = cr.child_id.astype(str)
cr["noun_text"] = cr.text.map(noun_text)
cr = cr[cr.noun_text.str.len() > 0]                       # keep pairs with >=1 noun
print(f"\naligned (clip>0.24, excl {HELD}, >=1 noun): {len(cr)}")

ridx = pd.read_parquet(W / "emb_reg/index.parquet")
have = set(zip(ridx.video_id, ridx.frame_idx.astype(int)))
cr = cr[[(v, int(f)) in have for v, f in zip(cr.video_id, cr.frame_idx)]]
print(f"with region embedding: {len(cr)}  (videos={cr.video_id.nunique()}, children={cr.child_id.nunique()})")

base = cr[["video_id", "frame_idx", "clip_score_max", "child_id"]].reset_index(drop=True)
full = base.copy(); full["text"] = cr.text.values
noun = base.copy(); noun["text"] = cr.noun_text.values
full.to_parquet(W / "manifests/aligned_full.parquet")
noun.to_parquet(W / "manifests/aligned_nouns.parquet")
print("wrote aligned_full / aligned_nouns (identical pairs)")
for i in [0, 1, 2]:
    print(f"  full: {full.text.iloc[i]!r}\n  noun: {noun.text.iloc[i]!r}")
print("mean tokens  full:", round(full.text.map(lambda t: len(_tok.findall(t.lower()))).mean(), 2),
      " noun:", round(noun.text.map(lambda t: len(t.split())).mean(), 2))
