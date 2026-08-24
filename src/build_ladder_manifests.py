"""Build the four ladder manifests + scaling + diversity subsets for any release, from a Gemini
scored parquet. Consolidates logic that was split across build_grid_manifests.py (baseline, t2)
and build_phase2.py (t15_filtnat, t15), which only ever worked for 2025.2.

Ladder rungs (each is a training manifest; the model and eval never change):
  base      all pairs, natural utterance                  -> pure / region-MIL
  filtnat   referent-bearing pairs, natural utterance     -> + alignment filter
  t15       referent-bearing, referent word WHERE SPOKEN  -> + word selection
  t2        referent-bearing, text = the referent label   -> + vision binding (clean-label ceiling)

usage: python src/build_ladder_manifests.py --scored scored/bv2026_gemini.parquet \
         --prefix bv26 [--english-filter manifests/bv26_en.parquet] \
         --sizes 10000,30000,100000,300000,1000000 --kids 1,3,10,25,51
"""
import argparse, re
import numpy as np, pandas as pd
from common import tokenize

ap = argparse.ArgumentParser()
ap.add_argument("--scored", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--english-filter", default="", help="pair manifest to intersect with (video-level English rule)")
ap.add_argument("--sizes", default="3000,10000,30000,100000,300000,1000000")
ap.add_argument("--index", default="", help="release_index.tsv: keep only these videos (post-release exclusions)")
ap.add_argument("--ladder-scales", default="", help="also build all 4 rungs from a matched subsample at these sizes")
ap.add_argument("--aligned-sizes", default="10000,30000,85000,150000")
ap.add_argument("--kids", default="1,3,10,25,50")
ap.add_argument("--div-n", type=int, default=30000, help="fixed pair count for the diversity sweep")
ap.add_argument("--seeds", default="0,1,2", help="one SUBSAMPLE per seed (not just one init)")
ap.add_argument("--small-seeds", default="0,1,2,3,4,5,6,7", help="extra seeds for sizes <= --small-max")
ap.add_argument("--mid-seeds", default="0,1,2,3,4", help="seeds for small-max < size <= --mid-max")
ap.add_argument("--mid-max", type=int, default=400000)
ap.add_argument("--small-max", type=int, default=100000)
a = ap.parse_args()
P, COLS = a.prefix, ["video_id", "frame_idx", "text"]


def sing(w):
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", w): return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w


G = pd.read_parquet(a.scored)
G = G[G.alignment.notna()].copy()
G["child"] = G.video_id.str.split("_").str[0]
print(f"scored pairs {len(G):,} | children {G.child.nunique()} | videos {G.video_id.nunique():,}")

if a.index:   # post-release exclusions (e.g. the mislabeled S02 subject)
    keep_v = set(pd.read_csv(a.index, sep="\t").video_id)
    n0 = len(G); G = G[G.video_id.isin(keep_v)]
    print(f"release_index filter: {n0:,} -> {len(G):,} pairs, {G.child.nunique()} children")

if a.english_filter:
    # intersect on the FULL pair key — a (video_id, text) join duplicates rows when a text
    # repeats within a video (the join-key failure class, again)
    keep = pd.read_parquet(a.english_filter)[["video_id", "frame_idx", "text"]].drop_duplicates()
    n0, k0 = len(G), G.child.nunique()
    G = G.merge(keep, on=["video_id", "frame_idx", "text"], how="inner")
    assert len(G) <= n0, "English intersect grew the manifest — join key is wrong"
    print(f"English filter: {n0:,} -> {len(G):,} pairs ({100*len(G)/n0:.1f}%), "
          f"children {k0} -> {G.child.nunique()}")

# ---- ladder ---------------------------------------------------------------------
G[COLS].to_parquet(f"manifests/{P}_base.parquet", index=False)
print(f"  base     {len(G):,}")
ref = G[G.referent.fillna("").str.len() > 0].copy()
ref[COLS].to_parquet(f"manifests/{P}_filtnat.parquet", index=False)
print(f"  filtnat  {len(ref):,}")

def t15_text(row):
    r = sing(str(row.referent).strip().lower())
    toks = set(tokenize(row.text)); toks |= {sing(t) for t in toks}
    return r if r in toks else row.text

r15 = ref.copy(); r15["text"] = r15.apply(t15_text, axis=1)
r15[COLS].to_parquet(f"manifests/{P}_t15.parquet", index=False)
print(f"  t15      {len(r15):,}  ({100*(r15.text.values != ref.text.values).mean():.0f}% relabeled)")

t2 = ref.copy(); t2["text"] = t2.referent.map(lambda x: sing(str(x).strip().lower()))
t2[COLS].to_parquet(f"manifests/{P}_t2.parquet", index=False)
print(f"  t2       {len(t2):,}  ({t2.text.nunique():,} distinct labels)")

# ---- scaling: random, ONE SUBSAMPLE PER SEED ------------------------------------
# Previously a single manifest was reused by every seed, so the error bars measured only
# initialisation and batch order. At 10k of 1.8M, WHICH 10k you draw is plausibly the larger
# source of variance, and it was invisible. Drawing per seed captures both at no extra compute.
SEEDS = [int(x) for x in a.seeds.split(",")]
SMALL = [int(x) for x in a.small_seeds.split(",")]
for N in [int(x) for x in a.sizes.split(",")]:
    if N > len(G):
        print(f"  skip rand_{N} (only {len(G):,} pairs)"); continue
    MID = [int(x) for x in a.mid_seeds.split(",")]
    # small N: draws nearly independent -> subsample variance dominates -> more of them.
    # at 1M of 1.8M any two draws share >50% of pairs, so extra draws buy little.
    seeds = SMALL if N <= a.small_max else (MID if N <= a.mid_max else SEEDS)
    for s in seeds:
        G.sample(N, random_state=1000 + s)[COLS].to_parquet(
            f"manifests/{P}_rand_{N}_s{s}.parquet", index=False)
    print(f"  rand_{N}: {len(seeds)} independent subsamples (seeds {seeds})")

# ---- scaling: aligned. SHUFFLE first — alignment is an integer with huge tie groups,
#      so a plain head(N) cuts inside a tie in file order (i.e. by video and child).
ma = (G.sample(frac=1.0, random_state=0)
        .sort_values("alignment", ascending=False, kind="stable"))
for N in [int(x) for x in a.aligned_sizes.split(",") if x]:
    if N > len(ma): print(f"  skip align_{N}"); continue
    s = ma.head(N)
    s[COLS].to_parquet(f"manifests/{P}_align_{N}.parquet", index=False)
    print(f"  align_{N}: min alignment {s.alignment.min():.0f}, {s.child.nunique()} children")

# ---- ladder at reduced scale: rung separations off the full-corpus ceiling ------
# Within a (scale, seed) cell all four rungs derive from the SAME base subsample, so rung
# differences are never confounded with which pairs were drawn.
for N in [int(x) for x in a.ladder_scales.split(",") if x]:
    for s_ in SEEDS:
        sub = G.sample(min(N, len(G)), random_state=3000 + s_)
        sub[COLS].to_parquet(f"manifests/{P}_lad{N}_base_s{s_}.parquet", index=False)
        rsub = sub[sub.referent.fillna("").str.len() > 0].copy()
        rsub[COLS].to_parquet(f"manifests/{P}_lad{N}_filtnat_s{s_}.parquet", index=False)
        r15s = rsub.copy(); r15s["text"] = r15s.apply(t15_text, axis=1)
        r15s[COLS].to_parquet(f"manifests/{P}_lad{N}_t15_s{s_}.parquet", index=False)
        t2s = rsub.copy(); t2s["text"] = t2s.referent.map(lambda x: sing(str(x).strip().lower()))
        t2s[COLS].to_parquet(f"manifests/{P}_lad{N}_t2_s{s_}.parquet", index=False)
    print(f"  lad{N}: 4 rungs x {len(SEEDS)} matched subsamples (filtnat ~{len(rsub):,} of {len(sub):,})")

# ---- diversity at fixed count, RANDOM children per seed -------------------------
# Previously this took the k BIGGEST children deterministically: no child-draw variance, and
# biased toward large contributors (one child can be 7% of the corpus). Draw k at random per seed,
# preferring children with enough pairs to fill their share.
pc = G.groupby("child").size().sort_values(ascending=False)
N = a.div_n
DIV_SEEDS = [int(x) for x in a.mid_seeds.split(",")]     # 5 draws: child-draw variance is the point
for k in [int(x) for x in a.kids.split(",")]:
    for s in DIV_SEEDS:
        rng = np.random.default_rng(2000 + s)
        if k >= len(pc):
            kids = list(pc.index)
        else:
            # sample among children who could contribute their share, else fall back to all
            per_need = N // k
            elig = list(pc[pc >= min(per_need, pc.max())].index) or list(pc.index)
            pool = elig if len(elig) >= k else list(pc.index)
            kids = list(rng.choice(pool, k, replace=False))
        per = max(1, N // len(kids))
        parts = [G[G.child == c].sample(min(per, int(pc[c])), random_state=s) for c in kids]
        sub = pd.concat(parts)
        if len(sub) < N:
            extra = G[G.child.isin(kids)].drop(sub.index)
            if len(extra):
                sub = pd.concat([sub, extra.sample(min(N - len(sub), len(extra)), random_state=s)])
        sub[COLS].to_parquet(f"manifests/{P}_div_{k}c_s{s}.parquet", index=False)
    print(f"  div_{k}c: {len(DIV_SEEDS)} random child draws of {k}")
