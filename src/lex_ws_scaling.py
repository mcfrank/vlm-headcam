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
# encoder -> (scaling family prefix, full-corpus family), all on the same training draws
# family -> (scaling-run pattern, full-corpus run, word2vec text stem pattern).
# F_* are the FINAL audio-filtered corpus (bv26a manifests) and are what the paper uses;
# the B26_/C8_ preview families are kept so the earlier figures still rebuild.
FAMILIES = {
    "F-dinov3b": ("F_dinov3b_rand_{n}", "F_dinov3b_base", "bv26a"),
    "F-dinov3l": ("F_dinov3l_rand_{n}", "F_dinov3l_base", "bv26a"),
    "F-vitb_bv": ("F_vitb_bv_rand_{n}", "F_vitb_bv_base", "bv26a"),
    "F-vits_bv": ("F_vits_bv_rand_{n}", "F_vits_bv_base", "bv26a"),
    "B26": ("B26_rand_{n}", "B26_lad_base", "bv26"),
    "L-OTS": ("C8_dinov3l_grid4x4_rand_{n}", "C8_dinov3l_grid4x4_base", "bv26"),
    "L-BV": ("C8_dinov3l_bv_grid4x4_rand_{n}", "C8_dinov3l_bv_grid4x4_base", "bv26"),
}
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
FAM = "B26"


def main():
    global FAM
    rows = []
    for sc in SCALES:
        pre, full_fam, corpus = FAMILIES[FAM]
        wp = cache / "w2v" / (f"{corpus}_rand_{sc}_s0.npz" if sc != "full"
                              else f"{corpus}_base.npz")
        if not wp.exists():
            continue
        v2, W2 = LS.load_npz(wp)
        model_glob = (pre.format(n=sc) if sc != "full" else full_fam) + "_s*.npz"
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
    out.to_csv(R / "results" / f"lexicon_ws_scaling_{FAM}.csv", index=False)
    print(out.groupby(["scale", "kind", "category"]).spearman.mean().round(3).to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="B26")
    FAM = ap.parse_args().family
    main()
