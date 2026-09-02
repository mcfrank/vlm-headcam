"""IN-DOMAIN held-out evaluation set (SI): word -> 4 BabyView frames, from the 1,258
language-excluded videos — same visual domain, never in word-learning training, Gemini
referent annotations and all four encoders' embeddings already exist.

Construction: pairs with alignment >= --min-align and a referent; category = singularized
referent; ONE frame per video per category (near-duplicate control); embedding dedupe
(dinov3b cos > 0.92 within category dropped); categories need >= --min-ex exemplars from
distinct videos. Emits the eval manifest + a 200-item human-validation sample list.
usage: python src/build_indomain_eval.py"""
import argparse
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import frame_key

O = "/ccn2b/dataset/babyview/2026.1/outputs"
ap = argparse.ArgumentParser()
ap.add_argument("--min-align", type=int, default=100)
ap.add_argument("--min-ex", type=int, default=4)
ap.add_argument("--max-ex", type=int, default=20)
ap.add_argument("--out", default="manifests/eval_frames_indomain.parquet")
a = ap.parse_args()


def sing(w):
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", w): return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w

dec = pd.read_csv(f"{O}/annotations/language/audio/video_decisions_audio_2026.1.csv")
held = set(dec.loc[dec.decision == "drop-non-english", "video_id"])
G = pd.read_parquet(f"{O}/annotations/referent/gemini_2026.1.parquet")
g = G[G.video_id.isin(held) & (G.alignment >= a.min_align) & (G.referent.fillna("").str.len() > 0)].copy()
g["category"] = g.referent.str.strip().str.lower().map(sing)
g = g[g.category.str.match(r"^[a-z]{2,}$")]
# Exclude person-terms, body parts, and generic containers: in egocentric video these appear
# in most frames, so 4AFC foils would contain the target class and the item measures nothing.
BAD = {"baby", "mom", "mommy", "mama", "dad", "daddy", "papa", "person", "people", "boy",
       "girl", "man", "woman", "kid", "child", "hand", "foot", "feet", "face", "head", "hair",
       "finger", "toe", "eye", "nose", "mouth", "ear", "arm", "leg", "tummy", "belly",
       "toy", "thing", "stuff", "one", "food", "room", "floor", "wall", "house", "home"}
g = g[~g.category.isin(BAD)]
print(f"held-out videos with candidates: {g.video_id.nunique():,} | candidate pairs {len(g):,} "
      f"| raw categories {g.category.nunique():,}")

# one frame per (video, category): the exemplar with max alignment then earliest frame
g = (g.sort_values(["alignment", "frame_idx"], ascending=[False, True])
       .drop_duplicates(["video_id", "category"]))

# embedding dedupe within category (dinov3b mean-region vectors)
emb = np.load(f"{O}/image_embeddings/dinov3b_grid4x4/emb.f16.npy", mmap_mode="r")
ix = pd.read_parquet(f"{O}/image_embeddings/dinov3b_grid4x4/index.parquet")
lut = {frame_key(v, int(f)): int(r) for v, f, r in zip(ix.video_id, ix.frame_idx, ix.row)}
rows_out = []
rng = np.random.default_rng(0)
for cat, grp in g.groupby("category"):
    grp = grp.sample(frac=1.0, random_state=0)
    keep, vecs = [], []
    for r in grp.itertuples():
        k = lut.get(frame_key(r.video_id, int(r.frame_idx)))
        if k is None: continue
        v = emb[k].astype(np.float32).mean(0)
        v /= np.linalg.norm(v) + 1e-9
        if any(float(v @ u) > 0.92 for u in vecs): continue
        keep.append(r); vecs.append(v)
        if len(keep) >= a.max_ex: break
    if len(keep) >= a.min_ex:
        for r in keep:
            rows_out.append(dict(video_id=r.video_id, frame_idx=int(r.frame_idx),
                                 category=cat, alignment=int(r.alignment)))
ev = pd.DataFrame(rows_out)
print(f"FINAL: {len(ev):,} frames | {ev.category.nunique()} categories | "
      f"{ev.video_id.nunique()} videos | exemplars/cat median {ev.groupby('category').size().median():.0f}")
ev.to_parquet(a.out, index=False)
val = ev.sample(min(200, len(ev)), random_state=1)
val.to_csv("scratch/indomain_validation_sample.csv", index=False)
print(f"wrote {a.out} + scratch/indomain_validation_sample.csv (human spot-check list)")
print("top categories:", ", ".join(ev.category.value_counts().head(12).index))
