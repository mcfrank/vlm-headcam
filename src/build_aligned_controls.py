"""Supplementary controls for the alignment-selection claim (reviewer critique):
  alignedonly   the aligned set A (alignment >= 50), |A| = ~172k
  matchrand     random NON-aligned pairs, |A|, matched to A's joint distribution of
                (primary eval-noun x utterance-length bin) — equates evaluation-noun
                exposure and the short/deictic profile that alignment selection enriches
  minusaligned  full corpus minus A            (does removing 10% collapse learning?)
  minusrand     full corpus minus random |A|   (…or is it just 10% less data?)
Emits manifests + results/aligned_control_diagnostics.json (what the matching equated).
usage: python src/build_aligned_controls.py"""
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import tokenize

O = "/ccn2b/dataset/babyview/2026.1/outputs"
COLS = ["video_id", "frame_idx", "text"]


def sing(w):
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", w): return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w

en = pd.read_parquet("manifests/bv26_pairs_en_audio.parquet")
G = pd.read_parquet(f"{O}/annotations/referent/gemini_2026.1.parquet", columns=COLS + ["alignment"])
G = G.drop_duplicates(COLS)          # the pair key can repeat (same text, same second); no fan-out
en = en.merge(G, on=COLS, how="left")
assert len(en) == 1_686_105, f"corpus fan-out: {len(en):,}"
evalwords = set(sing(c) for c in pd.concat([pd.read_parquet("manifests/eval_frames_konkle.parquet").category,
                                            pd.read_parquet("manifests/eval_frames_konkle_dev.parquet").category]).unique())
toks = en.text.map(lambda t: [sing(x) for x in tokenize(t)])
en["nw"] = toks.map(len)
en["lenbin"] = pd.cut(en.nw, [0, 2, 5, 10, 999], labels=["1-2", "3-5", "6-10", "11+"])
en["evalw"] = toks.map(lambda ts: next((t for t in ts if t in evalwords), "none"))
A = en[en.alignment >= 50]
pool = en[~(en.alignment >= 50)]
n = len(A)
print(f"corpus {len(en):,} | aligned A {n:,} ({100*n/len(en):.1f}%) | pool {len(pool):,}")

A[COLS].to_parquet("manifests/bv26a_alignedonly.parquet", index=False)
pool[COLS].to_parquet("manifests/bv26a_minusaligned.parquet", index=False)
diag = {"n_aligned": n, "n_corpus": len(en)}
for s in range(3):
    rng = np.random.default_rng(6000 + s)
    # minus-random: drop a random |A|
    drop = rng.choice(len(en), n, replace=False)
    en.drop(en.index[drop])[COLS].to_parquet(f"manifests/bv26a_minusrand_s{s}.parquet", index=False)
    # matched random: per (evalw, lenbin) cell, sample A's count from the pool
    target = A.groupby(["evalw", "lenbin"], observed=True).size()
    parts, deficit = [], 0
    for (w, lb), c in target.items():
        cell = pool[(pool.evalw == w) & (pool.lenbin == lb)]
        take = min(c, len(cell)); deficit += c - take
        if take: parts.append(cell.sample(take, random_state=int(rng.integers(1e9))))
    m = pd.concat(parts)
    if deficit:   # fill from the 'none' cells to keep |match| == |A|
        rest = pool.drop(m.index); rest = rest[rest.evalw == "none"]
        m = pd.concat([m, rest.sample(deficit, random_state=s)])
    m[COLS].to_parquet(f"manifests/bv26a_matchrand_s{s}.parquet", index=False)
    if s == 0:
        ev_A = int((A.evalw != "none").sum()); ev_m = int((m.evalw != "none").sum())
        rnd = en.sample(n, random_state=0); ev_r = int((rnd.evalw != "none").sum())
        diag.update(dict(evalnoun_pairs_aligned=ev_A, evalnoun_pairs_matched=ev_m,
                         evalnoun_pairs_plainrandom=ev_r, cell_deficit=int(deficit),
                         meanlen_aligned=round(float(A.nw.mean()), 2),
                         meanlen_matched=round(float(m.nw.mean()), 2),
                         meanlen_plainrandom=round(float(rnd.nw.mean()), 2)))
json.dump(diag, open("results/aligned_control_diagnostics.json", "w"), indent=2)
print(json.dumps(diag, indent=1))
