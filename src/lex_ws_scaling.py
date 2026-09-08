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
# family -> (run pattern, full-corpus run or None, w2v corpus stem, scale ladder or None).
# The ALIGNED families are the oracle arm: every pair is a Gemini-aligned referential
# moment, so they run out of data at ~170k and their w2v topline is trained on the SAME
# aligned utterances (bv26a_align_<n>), keeping the language-only control matched.
FREE_SCALES = [3000, 10000, 30000, 100000, 300000, 1000000, "full"]
ALIGN_SCALES = [10000, 30000, 100000, 170000]
FAMILIES = {
    "F-dinov3b": ("F_dinov3b_rand_{n}", "F_dinov3b_base", "bv26a", None),
    "F-dinov3l": ("F_dinov3l_rand_{n}", "F_dinov3l_base", "bv26a", None),
    "F-vitb_bv": ("F_vitb_bv_rand_{n}", "F_vitb_bv_base", "bv26a", None),
    "F-vits_bv": ("F_vits_bv_rand_{n}", "F_vits_bv_base", "bv26a", None),
    "F-dinov3b-aligned": ("F_dinov3b_align_{n}", None, "bv26a_align", ALIGN_SCALES),
    "F-dinov3l-aligned": ("F_dinov3l_align_{n}", None, "bv26a_align", ALIGN_SCALES),
    "F-vitb_bv-aligned": ("F_vitb_bv_align_{n}", None, "bv26a_align", ALIGN_SCALES),
    "F-vitl_bv": ("F_vitl_bv_rand_{n}", "F_vitl_bv_base", "bv26a", None),
    "F-vitl_bv-aligned": ("F_vitl_bv_align_{n}", None, "bv26a_align", ALIGN_SCALES),
    "F-dinov3s": ("F_dinov3s_rand_{n}", "F_dinov3s_base", "bv26a", None),
    "F-dinov3s-aligned": ("F_dinov3s_align_{n}", None, "bv26a_align", ALIGN_SCALES),
    "F-vits_bv-aligned": ("F_vits_bv_align_{n}", None, "bv26a_align", ALIGN_SCALES),
    "B26": ("B26_rand_{n}", "B26_lad_base", "bv26", None),
    "L-OTS": ("C8_dinov3l_grid4x4_rand_{n}", "C8_dinov3l_grid4x4_base", "bv26", None),
    "L-BV": ("C8_dinov3l_bv_grid4x4_rand_{n}", "C8_dinov3l_bv_grid4x4_base", "bv26", None),
}


def w2v_path(cache, corpus, sc):
    """The language-only topline trained on the SAME utterances as the model."""
    if corpus.endswith("_align"):
        return cache / "w2v" / f"{corpus}_{sc}.npz"
    return cache / "w2v" / (f"{corpus}_rand_{sc}_s0.npz" if sc != "full"
                            else f"{corpus}_base.npz")
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

SCALES = FREE_SCALES
FAM = "B26"


def main():
    global FAM
    rows = []
    pre, full_fam, corpus, scales = FAMILIES[FAM]
    for sc in (scales or SCALES):
        wp = w2v_path(cache, corpus, sc)
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
                n = (1_682_259 if str(FAM).startswith("F-") else 1_815_243) \
                    if sc == "full" else sc
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
