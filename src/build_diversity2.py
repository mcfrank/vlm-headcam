"""Diversity sweep at higher pair floors (the 30k floor was low enough that a single child
covers it, which under-expresses what diversity could contribute). k random children at a
fixed total of N pairs, one independent child-draw per seed. A (k, N) cell whose top-k
children cannot jointly supply N pairs is SKIPPED loudly, never silently underfilled.

usage: python src/build_diversity2.py --scored .../gemini_2026.1.parquet \
    --english-filter manifests/bv26_pairs_en.parquet --index .../release_index.tsv
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scored", required=True)
ap.add_argument("--english-filter", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--prefix", default="bv26")
ap.add_argument("--specs", default="100000:1,3,10,25,50;300000:10,25,50")
ap.add_argument("--seeds", default="0,1,2,3,4")
a = ap.parse_args()
COLS = ["video_id", "frame_idx", "text"]

G = pd.read_parquet(a.scored)
G = G[G.alignment.notna()].copy()
G["child"] = G.video_id.str.split("_").str[0]
keep_v = set(pd.read_csv(a.index, sep="\t").video_id)
G = G[G.video_id.isin(keep_v)]
keep = pd.read_parquet(a.english_filter)[["video_id", "frame_idx", "text"]].drop_duplicates()
G = G.merge(keep, on=["video_id", "frame_idx", "text"], how="inner")
pc = G.groupby("child").size().sort_values(ascending=False)
print(f"corpus {len(G):,} pairs, {len(pc)} children | top child {pc.iloc[0]:,} | median {int(pc.median()):,}")

SEEDS = [int(x) for x in a.seeds.split(",")]
for spec in a.specs.split(";"):
    N, ks = spec.split(":")
    N = int(N)
    for k in [int(x) for x in ks.split(",")]:
        if pc.nlargest(k).sum() < N:
            print(f"  SKIP div{N//1000}k_{k}c: top-{k} children hold only {pc.nlargest(k).sum():,} < {N:,}")
            continue
        per_need = N // k
        elig = list(pc[pc >= per_need].index)
        for s in SEEDS:
            rng = np.random.default_rng(4000 + s)
            if len(elig) >= k:
                kids = list(rng.choice(elig, k, replace=False))
            else:
                # not enough children can supply an equal share: take all eligible, fill the
                # remaining slots with the largest others (documented bias, printed below)
                kids = elig + list(pc.drop(elig).nlargest(k - len(elig)).index)
            sub_parts = []
            budget = N
            for j, c in enumerate(sorted(kids, key=lambda c: pc[c])):
                take = min(int(pc[c]), budget // (k - j))
                sub_parts.append(G[G.child == c].sample(take, random_state=s))
                budget -= take
            sub = pd.concat(sub_parts)
            if len(sub) < N:      # distribute the shortfall over remaining capacity
                extra = G[G.child.isin(kids)].drop(sub.index)
                sub = pd.concat([sub, extra.sample(min(N - len(sub), len(extra)), random_state=s)])
            assert len(sub) == N, f"underfilled {len(sub):,} < {N:,}"
            sub[COLS].to_parquet(f"manifests/{a.prefix}_div{N//1000}k_{k}c_s{s}.parquet", index=False)
        print(f"  div{N//1000}k_{k}c: {len(SEEDS)} draws | eligible children {len(elig)}"
              + ("" if len(elig) >= k else f" (< k: filled from largest)"))
