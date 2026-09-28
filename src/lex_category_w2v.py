"""CDI-category cohesion: the grounded model vs word2vec trained on the same utterances.

Raw cosine gaps are not comparable across the two spaces (on shared words the model's mean
pairwise cosine is ~0.01, word2vec's ~0.18), so the statistic here is scale-free: per-category
AUC = P(within-category pair more similar than a member-to-nonmember pair); 0.5 = chance.
p-values from a per-category label-permutation null, as in lex_category_structure.py.

Two item sets, reported separately:
  shared  all CDI object-category words present in BOTH vocabularies (model-independent)
  fig3b   the model's above-null CDI nouns (fig3B's set; selected by the model, so it
          favours the model in a head-to-head -- kept for continuity only)

-> results/lexicon_category_w2v_<family>.csv, one row per (item_set, source, category).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = Path(__file__).resolve().parent.parent
CATS = ["animals", "food_drink", "vehicles", "toys", "clothing", "body_parts",
        "household", "furniture_rooms", "outside", "places"]
MIN_N = 4

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="F_dinov3l_base")
ap.add_argument("--w2v", default="bv26a_base", help="word2vec on the same corpus")
ap.add_argument("--perms", type=int, default=2000)
a = ap.parse_args()

C = R / "._lexicon_cache"
cdi = pd.read_csv(R / "results" / "cdi_categories.csv").set_index("word").category
sources = [(p.stem, p) for p in sorted((C / "emb").glob(f"{a.family}_s*.npz"))]
sources.append((f"w2v:{a.w2v}", C / "w2v" / f"{a.w2v}.npz"))


def load(p):
    z = np.load(p, allow_pickle=True)
    return {str(w): i for i, w in enumerate(z["words"])}, z["W"].astype(np.float32)


spaces = {name: load(p) for name, p in sources}
vocab = set.intersection(*(set(ix) for ix, _ in spaces.values()))
shared = [w for w, c in cdi.items() if c in CATS and w in vocab]
t = pd.read_csv(R / "results" / f"lexicon_tsne_{sources[0][0]}_NOUN.csv")
fig3b = [w for w in t.word if cdi.get(w) in CATS and w in vocab]


def aucs(S, lab, cats):
    """-> {category: AUC of within- vs between-category pair similarities}."""
    out = {}
    for c in cats:
        m = lab == c
        n = int(m.sum())
        w = S[np.ix_(m, m)][np.triu_indices(n, 1)]
        b = S[np.ix_(m, ~m)].ravel()
        r = rankdata(np.concatenate([w, b]))
        out[c] = (r[:len(w)].sum() - len(w) * (len(w) + 1) / 2) / (len(w) * len(b))
    return out


rows = []
for set_name, words in [("shared", shared), ("fig3b", fig3b)]:
    lab = np.array([cdi[w] for w in words])
    cats = [c for c in CATS if (lab == c).sum() >= MIN_N]
    for name, (ix, W) in spaces.items():
        E = W[[ix[w] for w in words]]
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
        S = E @ E.T
        obs = aucs(S, lab, cats)
        rng = np.random.default_rng(0)
        null = {c: np.empty(a.perms) for c in cats}
        for k in range(a.perms):
            for c, v in aucs(S, rng.permutation(lab), cats).items():
                null[c][k] = v
        for c in cats:
            rows.append(dict(item_set=set_name, source=name,
                             kind="w2v" if name.startswith("w2v") else "model",
                             category=c, n=int((lab == c).sum()), auc=obs[c],
                             null_lo=np.percentile(null[c], 2.5),
                             null_hi=np.percentile(null[c], 97.5),
                             p_perm=(np.sum(null[c] >= obs[c]) + 1) / (a.perms + 1)))
        print(f"  {set_name:6s} {name:24s} done")

out = pd.DataFrame(rows)
dst = R / "results" / f"lexicon_category_w2v_{a.family}.csv"
out.to_csv(dst, index=False)
print(f"wrote {dst.relative_to(R)}: shared n={len(shared)}, fig3b n={len(fig3b)}")
