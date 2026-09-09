"""Stratified sample of (frame, utterance) pairs for the human check of the Gemini alignment
annotation. Runs on ccn2-14 (needs the final corpus manifest + the scored table).

Population = final training corpus (manifests/bv26_pairs_en_audio.parquet) joined to the Gemini
table on (video_id, frame_idx, text), deduped on that triple, scored (no error), 2-20 tokens.
Strata partition the population by Gemini score so every response can be reweighted to
corpus-level precision/recall of the >=50 rule:

    0_noun  : alignment 0 and the utterance contains a concrete object noun (hard negatives)
    0_other : alignment 0, no concrete noun
    1-49    : rare in-between scores (10/20/30/40)
    50-60, 70-75, 80, 90-95, 100 : the aligned bins

Constraints: <=30 items per child, <=1 item per (video, minute), <=1 item per (video, stratum).
Items get opaque random ids so the app never exposes video/child ids.

Usage (node):
    cd /data2/mcfrank/vlm-headcam && /data2/mcfrank/gemini_check/venv/bin/python \
        human_check/sample.py --out /data2/mcfrank/gemini_check/sample.parquet
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GEMINI = "/ccn2b/dataset/babyview/2026.1/outputs/annotations/referent/gemini_2026.1.parquet"
MANIFEST = "manifests/bv26_pairs_en_audio.parquet"
KONKLE = ["manifests/eval_frames_konkle.parquet", "manifests/eval_frames_konkle_dev.parquet"]
CDI = Path(__file__).resolve().parent.parent / "results" / "cdi_categories.csv"
CDI_OBJECT_CATS = {"animals", "food_drink", "household", "furniture_rooms", "toys", "clothing",
                   "body_parts", "vehicles", "outside"}
NOT_NOUNS = {"can"}  # CDI lists it (container) but in transcripts it is almost always the modal

# (name, lo, hi, n) — bins partition the score range 0..100
STRATA = [("0_noun", 0, 0, 200), ("0_other", 0, 0, 100), ("1-49", 1, 49, 50),
          ("50-60", 50, 60, 150), ("70-75", 70, 75, 150), ("80", 80, 80, 150),
          ("90-95", 90, 95, 100), ("100", 100, 100, 150)]
MAX_PER_CHILD = 30


def concrete_nouns():
    words = set()
    for p in KONKLE:
        if Path(p).exists():
            words |= set(pd.read_parquet(p).category.str.lower())
    cdi = pd.read_csv(CDI)
    for w in cdi[cdi.category.isin(CDI_OBJECT_CATS)].word:
        w = re.sub(r"\(.*?\)", "", str(w)).strip().lower()
        if w:
            words.add(w)
    return words - NOT_NOUNS


def singular(t):
    if t.endswith("ies"):
        return t[:-3] + "y"
    if t.endswith("es"):
        return t[:-2]
    if t.endswith("s"):
        return t[:-1]
    return t


def noun_hits(text, nouns):
    toks = re.sub(r"[^a-z' ]", " ", text.lower()).split()
    hits = {t for t in toks if t in nouns or singular(t) in nouns}
    padded = " " + " ".join(toks) + " "
    hits |= {w for w in nouns if " " in w and f" {w} " in padded}
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    m = pd.read_parquet(MANIFEST).drop_duplicates(["video_id", "frame_idx", "text"])
    g = pd.read_parquet(GEMINI).drop_duplicates(["video_id", "frame_idx", "text"])
    d = m.merge(g[["video_id", "frame_idx", "text", "alignment", "referent", "error"]],
                on=["video_id", "frame_idx", "text"], how="inner")
    d = d[d.alignment.notna() & d.error.isna()].copy()
    d["alignment"] = d.alignment.astype(int)
    d["referent"] = d.referent.fillna("")
    n_all = len(d)
    ntok = d.text.str.split().str.len()
    d = d[(ntok >= 2) & (ntok <= 20)].copy()
    print(f"population: {n_all:,} scored pairs; {len(d):,} with 2-20 tokens")

    nouns = concrete_nouns()
    print(f"{len(nouns)} concrete-noun forms")
    hits = d.text.map(lambda t: noun_hits(t, nouns))
    d["has_noun"] = hits.str.len() > 0
    d["noun_hits"] = hits.map(",".join)

    def stratum(a, has_noun):
        if a == 0:
            return "0_noun" if has_noun else "0_other"
        for name, lo, hi, _ in STRATA[2:]:
            if lo <= a <= hi:
                return name
        raise ValueError(a)
    d["stratum"] = [stratum(a, h) for a, h in zip(d.alignment, d.has_noun)]
    d["minute"] = d.frame_idx // 60
    pop = d.stratum.value_counts()

    # greedy round-robin draw under the constraints
    pools = {name: d[d.stratum == name].sample(frac=1, random_state=int(rng.integers(2**31)))
             for name, *_ in STRATA}
    cursors = {name: 0 for name in pools}
    picked = {name: [] for name in pools}
    per_child, used_min, used_vs = {}, set(), set()
    active = True
    while active:
        active = False
        for name, _, _, n in STRATA:
            if len(picked[name]) >= n:
                continue
            pool = pools[name]
            while cursors[name] < len(pool):
                r = pool.iloc[cursors[name]]
                cursors[name] += 1
                if per_child.get(r.child_id, 0) >= MAX_PER_CHILD:
                    continue
                if (r.video_id, r.minute) in used_min or (r.video_id, name) in used_vs:
                    continue
                per_child[r.child_id] = per_child.get(r.child_id, 0) + 1
                used_min.add((r.video_id, r.minute))
                used_vs.add((r.video_id, name))
                picked[name].append(r.name)
                active = True
                break
    rows = []
    for name, _, _, n in STRATA:
        if len(picked[name]) < n:
            print(f"WARNING stratum {name}: only {len(picked[name])}/{n} drawn under constraints")
        rows.extend(picked[name])
    s = d.loc[rows].copy()
    s["n_sample"] = s.stratum.map(s.stratum.value_counts())
    s["n_pop"] = s.stratum.map(pop)
    s["weight"] = s.n_pop / s.n_sample
    ids = set()
    while len(ids) < len(s):
        ids.add(f"{rng.integers(16**8):08x}")
    s["item_id"] = sorted(ids)
    s = s.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    s = s.rename(columns={"alignment": "gemini_alignment", "referent": "gemini_referent"})
    cols = ["item_id", "video_id", "utterance_id", "frame_idx", "text", "child_id", "speaker",
            "gemini_alignment", "gemini_referent", "has_noun", "noun_hits", "stratum", "n_pop",
            "n_sample", "weight"]
    s[cols].to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {len(s)} items, {s.child_id.nunique()} children, "
          f"max/child {s.child_id.value_counts().max()}")
    summ = (s.groupby("stratum").agg(n_sample=("item_id", "size"), n_pop=("n_pop", "first"),
                                     weight=("weight", "first"))
            .reindex([n for n, *_ in STRATA]))
    summ["pop_frac"] = summ.n_pop / summ.n_pop.sum()
    print(summ.to_string())
    summ.to_csv(Path(args.out).with_name("strata.csv"))


if __name__ == "__main__":
    main()
