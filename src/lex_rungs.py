"""The oracle ladder in lexical space: what each rung of referential supervision does to
the learned lexicon, at the ~170k pairs the aligned arm runs out at.

Rungs (all full aligned scale, so pair counts are matched):
  free-at-scale  the unfiltered arm subsampled to the same pair count (rand_100000 is the
                 closest free point; reported for reference, not matched exactly)
  aligned        every pair is a Gemini-aligned referential moment (align_170000)
  filtnat        the ladder's alignment-filter rung
  t15 / t2       + word selection / + vision binding (text becomes the referent label)

Each rung is scored against a word2vec trained on ITS OWN utterances, so the language-only
control degrades with the text the rung actually shows the model. -> results/lexicon_rungs.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import lex_score as LS
from lex_ws_scaling import pairs, cache

R = Path(__file__).resolve().parent.parent
ENCODERS = ["dinov3b", "dinov3l", "dinov3s", "vitb_bv", "vits_bv", "vitl_bv"]
RUNGS = [("aligned", "F_{e}_align_170000", "bv26a_align_170000"),
         ("filtnat", "F_{e}_lad_filtnat", "bv26a_filtnat"),
         ("word selection", "F_{e}_lad_t15", "bv26a_t15"),
         ("vision binding", "F_{e}_lad_t2", "bv26a_t2")]


def partial(a, b, c):
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
    res = [v - np.polyval(np.polyfit(rc, v, 1), rc) for v in (ra, rb)]
    return stats.pearsonr(res[0], res[1])[0]


rows = []
for e in ENCODERS:
    for label, pat, w2v_stem in RUNGS:
        wp = cache / "w2v" / f"{w2v_stem}.npz"
        if not wp.exists():
            continue
        v2, W2 = LS.load_npz(wp)
        for mp in sorted((cache / "emb").glob(pat.format(e=e) + "_s*.npz")):
            v1, W1 = LS.load_npz(mp)
            ok = pairs[pairs.w1.isin(v1) & pairs.w2.isin(v1)]
            both = ok[ok.w1.isin(v2) & ok.w2.isin(v2)]
            for cat, sub, sub2 in [("all", ok, both), ("noun", ok[ok.noun], both[both.noun])]:
                if len(sub) < 30:
                    continue
                s1 = [W1[v1[r.w1]] @ W1[v1[r.w2]] for r in sub.itertuples()]
                row = dict(encoder=e, rung=label, run=mp.stem, category=cat,
                           spearman=stats.spearmanr(s1, sub.z).correlation,
                           n_pairs=len(sub), model_vocab=len(v1))
                if len(sub2) >= 30:               # matched-pair comparison with its own w2v
                    m = [W1[v1[r.w1]] @ W1[v1[r.w2]] for r in sub2.itertuples()]
                    w = [W2[v2[r.w1]] @ W2[v2[r.w2]] for r in sub2.itertuples()]
                    row.update(spearman_shared=stats.spearmanr(m, sub2.z).correlation,
                               w2v_spearman=stats.spearmanr(w, sub2.z).correlation,
                               partial_model=partial(sub2.z.values, np.array(m), np.array(w)),
                               n_shared=len(sub2), w2v_vocab=len(v2))
                rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(R / "results" / "lexicon_rungs.csv", index=False)
n = out[out.category == "noun"]
print(n.groupby(["encoder", "rung"], sort=False)
       .agg(rho=("spearman", "mean"), shared=("spearman_shared", "mean"),
            w2v=("w2v_spearman", "mean"), partial=("partial_model", "mean"),
            vocab=("model_vocab", "first")).round(3).to_string())
