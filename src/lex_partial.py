"""Does vision add anything beyond distributional structure? Partial Spearman of
(human relatedness ~ two-tower similarity | word2vec similarity) on the shared pairs at
each scale — and the reverse direction for context. Rank-based: all three variables rank
transformed, partials via OLS residuals. -> results/lexicon_partial.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import lex_score as LS
import argparse
from lex_ws_scaling import pairs, SCALES, cache, FAMILIES  # pooled pairs + cache paths

R = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser(); _ap.add_argument("--family", default="B26")
FAM = _ap.parse_args().family


def partial(a, b, c):
    """corr(a, b | c) on ranks, with analytic p."""
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
    res = []
    for v in (ra, rb):
        beta = np.polyfit(rc, v, 1)
        res.append(v - np.polyval(beta, rc))
    r, _ = stats.pearsonr(res[0], res[1])
    n = len(a)
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), n - 3)


rows = []
for sc in SCALES:
    wp = cache / "w2v" / (f"bv26_rand_{sc}_s0.npz" if sc != "full" else "bv26_base.npz")
    v2, W2 = LS.load_npz(wp)
    pre, full_fam = FAMILIES[FAM]
    glob = (pre.format(n=sc) if sc != "full" else full_fam) + "_s*.npz"
    for mp in sorted((cache / "emb").glob(glob)):
        v1, W1 = LS.load_npz(mp)
        ok = pairs[pairs.w1.isin(v1) & pairs.w2.isin(v1) & pairs.w1.isin(v2) & pairs.w2.isin(v2)]
        for cat, sub in [("all", ok), ("noun", ok[ok.noun])]:
            if len(sub) < 30:
                continue
            s1 = np.array([W1[v1[r.w1]] @ W1[v1[r.w2]] for r in sub.itertuples()])
            s2 = np.array([W2[v2[r.w1]] @ W2[v2[r.w2]] for r in sub.itertuples()])
            h = sub.z.values
            r_m, p_m = partial(h, s1, s2)       # model beyond w2v
            r_w, p_w = partial(h, s2, s1)       # w2v beyond model
            n = 1815243 if sc == "full" else sc
            rows.append(dict(scale=n, run=mp.stem, category=cat, n_pairs=len(sub),
                             partial_model=r_m, p_model=p_m,
                             partial_w2v=r_w, p_w2v=p_w))
out = pd.DataFrame(rows)
out.to_csv(R / "results" / f"lexicon_partial_{FAM}.csv", index=False)
full = out[out.scale == 1815243].groupby("category")[["partial_model", "partial_w2v", "n_pairs"]].mean()
print("full corpus (mean over seeds):")
print(full.round(3).to_string())
print(out.groupby(["scale", "category"]).partial_model.mean().round(3).to_string())
