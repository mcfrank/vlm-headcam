"""ch6 scaling + whose-data manifests, all subsampled from the 911k train pairs (region features
already in emb_reg). A: random vs aligned scaling curves. B: diversity-at-fixed-count + within-
child ceiling."""
import numpy as np
import pandas as pd

G = (pd.read_parquet("scored/gemini_full.parquet")[["video_id", "frame_idx", "text", "alignment", "child_id"]]
     .drop_duplicates(["video_id", "frame_idx", "text"]))
tr = pd.read_parquet("manifests/grid_baseline_train.parquet")
m = tr.merge(G, on=["video_id", "frame_idx", "text"], how="left").dropna(subset=["child_id"]).reset_index(drop=True)
cols = ["video_id", "frame_idx", "text"]
print(f"train pool: {len(m)} pairs, {m.child_id.nunique()} children")

# A1 random (baseline) scaling
for N in [10000, 30000, 100000, 300000]:
    m.sample(N, random_state=0)[cols].to_parquet(f"manifests/scale_rand_{N}.parquet", index=False)

# A2 aligned (top-N Gemini) scaling
ma = m.sort_values("alignment", ascending=False)
for N in [10000, 30000, 85000]:
    ma.head(N)[cols].to_parquet(f"manifests/scale_align_{N}.parquet", index=False)

# B4 diversity at fixed count 30k: draw 30k from k biggest children (balanced)
pc = m.groupby("child_id").size().sort_values(ascending=False)
N = 30000
for k in [1, 3, 10, 36]:
    kids = list(pc.index[:k]) if k < 36 else list(pc.index)
    per = N // len(kids)
    parts = [m[m.child_id == kid].sample(min(per, int(pc[kid])), random_state=0) for kid in kids]
    sub = pd.concat(parts)
    if len(sub) < N:                                     # top up from the same kids
        extra = m[m.child_id.isin(kids) & ~m.index.isin(sub.index)]
        sub = pd.concat([sub, extra.sample(min(N - len(sub), len(extra)), random_state=1)])
    sub = sub.sample(min(N, len(sub)), random_state=0)
    sub[cols].to_parquet(f"manifests/scale_div_{k}c.parquet", index=False)
    print(f"  diversity {k}c: {len(sub)} pairs, {sub.child_id.nunique()} children")

# B6 within-child ceiling: biggest child (all) vs pooled random at matched count
big = pc.index[0]; Nb = int(pc.iloc[0])
m[m.child_id == big][cols].to_parquet("manifests/scale_bigchild.parquet", index=False)
m.sample(Nb, random_state=0)[cols].to_parquet("manifests/scale_poolbig.parquet", index=False)
print(f"  within-child: biggest child {big} = {Nb} pairs | pooled matched = {Nb}")
