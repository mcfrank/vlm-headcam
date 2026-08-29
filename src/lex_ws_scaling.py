"""Human word-relatedness across training scale: the two-tower lexicon vs the word2vec
topline trained on the same draws, on the SAME pairs (both vocabularies) at each scale.

Pairs pooled across the five relatedness sets (human scores z-scored within set) and split
by class: noun pairs (both words majority-NOUN in the corpus) vs all pairs.
-> results/lexicon_ws_scaling.csv (scale, kind, source, category, spearman, n_pairs)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lex_score as LS                      # reuse dataset parsing + npz loading

R = Path(__file__).resolve().parent.parent
cache = R / "._lexicon_cache"
pos = pd.read_csv(cache / "word_pos.csv").set_index("word")["pos"]

pairs = []
for name in LS.DATASETS:
    d = LS.read_pairs(name)
    d = d[d.w1 != d.w2].copy()
    d["z"] = (d.score - d.score.mean()) / d.score.std()
    d["dataset"] = name
    pairs.append(d[["w1", "w2", "z", "dataset"]])
pairs = pd.concat(pairs, ignore_index=True).drop_duplicates(["w1", "w2"])
pairs["noun"] = pairs.w1.map(pos).eq("NOUN") & pairs.w2.map(pos).eq("NOUN")
print(f"{len(pairs)} pooled pairs, {pairs.noun.sum()} noun-noun")

SCALES = [3000, 10000, 30000, 100000, 300000, 1000000, "full"]
rows = []
for sc in SCALES:
    wp = cache / "w2v" / (f"bv26_rand_{sc}_s0.npz" if sc != "full" else "bv26_base.npz")
    v2, W2 = LS.load_npz(wp)
    model_glob = f"B26_rand_{sc}_s*.npz" if sc != "full" else "B26_lad_base_s*.npz"
    for mp in sorted((cache / "emb").glob(model_glob)):
        v1, W1 = LS.load_npz(mp)
        ok = pairs[pairs.w1.isin(v1) & pairs.w2.isin(v1) & pairs.w1.isin(v2) & pairs.w2.isin(v2)]
        for cat, sub in [("all", ok), ("noun", ok[ok.noun])]:
            if len(sub) < 20:
                continue
            s1 = [W1[v1[r.w1]] @ W1[v1[r.w2]] for r in sub.itertuples()]
            s2 = [W2[v2[r.w1]] @ W2[v2[r.w2]] for r in sub.itertuples()]
            n = 1815243 if sc == "full" else sc
            rows.append(dict(scale=n, source=mp.stem, kind="model", category=cat,
                             spearman=spearmanr(s1, sub.z).correlation, n_pairs=len(sub)))
            rows.append(dict(scale=n, source=wp.stem, kind="w2v", category=cat,
                             spearman=spearmanr(s2, sub.z).correlation, n_pairs=len(sub)))
out = pd.DataFrame(rows).drop_duplicates(["scale", "source", "kind", "category"])
out.to_csv(R / "results" / "lexicon_ws_scaling.csv", index=False)
print(out.groupby(["scale", "kind", "category"]).spearman.mean().round(3).to_string())
