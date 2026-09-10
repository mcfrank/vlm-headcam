"""ch6 scaling + whose-data manifests, all subsampled from the 911k train pairs (region features
already in emb_reg). A: random vs aligned scaling curves. B: diversity-at-fixed-count + within-
child ceiling."""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scored", default="scored/gemini_full.parquet")
ap.add_argument("--pairs", default="manifests/grid_baseline_train.parquet")
ap.add_argument("--prefix", default="scale", help="manifests/<prefix>_rand_N.parquet etc")
ap.add_argument("--sizes", default="10000,30000,100000,300000")
ap.add_argument("--aligned-sizes", default="10000,30000,85000")
ap.add_argument("--kids", default="1,3,10,36")
A = ap.parse_args()
P = A.prefix

G = (pd.read_parquet(A.scored)[["video_id", "frame_idx", "text", "alignment", "child_id"]]
     .drop_duplicates(["video_id", "frame_idx", "text"]))
tr = pd.read_parquet(A.pairs)
m = tr.merge(G, on=["video_id", "frame_idx", "text"], how="left").dropna(subset=["child_id"]).reset_index(drop=True)
cols = ["video_id", "frame_idx", "text"]
print(f"train pool: {len(m)} pairs, {m.child_id.nunique()} children")

# A1 random (baseline) scaling
for N in [int(x) for x in A.sizes.split(",")]:
    if N > len(m): continue
    m.sample(N, random_state=0)[cols].to_parquet(f"manifests/{P}_rand_{N}.parquet", index=False)

# A2 aligned (top-N Gemini) scaling
# `alignment` is an INTEGER 0-100 and massively tied (24,677 pairs score exactly 100), so a
# plain head(N) cuts inside a tie group in file order — which is grouped by video and child, and
# gave the 10k arm only 24 of 36 children. Shuffle first, then a STABLE sort, so ties break at
# random. (src/build_gemini_arms.py already did this; this script did not.)
ma = (m.sample(frac=1.0, random_state=0)
        .sort_values("alignment", ascending=False, kind="stable"))
for N in [int(x) for x in A.aligned_sizes.split(",")]:
    if N > len(ma):
        print(f"  SKIP align_{N}: only {len(ma)} scored pairs available"); continue
    sub = ma.head(N)
    print(f"  align_{N}: min alignment {sub.alignment.min():.0f}, "
          f"{sub.child_id.nunique()} children")
    sub[cols].to_parquet(f"manifests/{P}_align_{N}.parquet", index=False)

# B4 diversity at fixed count 30k: draw 30k from k biggest children (balanced)
pc = m.groupby("child_id").size().sort_values(ascending=False)
N = 30000
for k in [int(x) for x in A.kids.split(",")]:
    kids = list(pc.index[:k]) if k < len(pc) else list(pc.index)
    per = N // len(kids)
    parts = [m[m.child_id == kid].sample(min(per, int(pc[kid])), random_state=0) for kid in kids]
    sub = pd.concat(parts)
    if len(sub) < N:                                     # top up from the same kids
        extra = m[m.child_id.isin(kids) & ~m.index.isin(sub.index)]
        sub = pd.concat([sub, extra.sample(min(N - len(sub), len(extra)), random_state=1)])
    sub = sub.sample(min(N, len(sub)), random_state=0)
    sub[cols].to_parquet(f"manifests/{P}_div_{k}c.parquet", index=False)
    print(f"  diversity {k}c: {len(sub)} pairs, {sub.child_id.nunique()} children")

# B6 within-child ceiling: biggest child (all) vs pooled random at matched count
big = pc.index[0]; Nb = int(pc.iloc[0])
m[m.child_id == big][cols].to_parquet(f"manifests/{P}_bigchild.parquet", index=False)
m.sample(Nb, random_state=0)[cols].to_parquet(f"manifests/{P}_poolbig.parquet", index=False)
print(f"  within-child: biggest child {big} = {Nb} pairs | pooled matched = {Nb}")
