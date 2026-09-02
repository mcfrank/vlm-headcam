"""CDI-category structure in the learned noun lexicon.

For each CDI object category, the statistic is that category's cohesion gap: the mean
cosine among its members minus the mean cosine from its members to every other noun, in
the full embedding space (not the t-SNE). Each category gets its OWN label-permutation
null -- shuffle the category labels across nouns and recompute that category's gap -- so
per-category claims are supported by per-category tests. The pooled gap and its null are
also reported, as one extra row per seed with category="ALL".

Run over every seed of a family so the figure can show across-seed variability. The noun
set is held fixed at the seed-0 above-null set (the seeds share a vocabulary, so this
isolates embedding variability rather than re-running the item filter per seed).

-> results/lexicon_category_structure_<family>.csv, one row per (seed, category).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
CATS = ["animals", "food_drink", "vehicles", "toys", "clothing", "body_parts",
        "household", "furniture_rooms", "outside", "places"]
MIN_N = 4

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="C8_dinov3l_grid4x4_base",
                help="run family; every <family>_s<seed> cache found is used")
ap.add_argument("--perms", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

EMB = R / "._lexicon_cache" / "emb"
runs = sorted(p.stem for p in EMB.glob(f"{a.family}_s*.npz"))
if not runs:
    raise SystemExit(f"no caches for {a.family} in {EMB}")

# item set: the seed-0 above-null nouns that carry a CDI object category
nouns = pd.read_csv(R / "results" / f"lexicon_tsne_{runs[0]}_NOUN.csv")
cdi = pd.read_csv(R / "results" / "cdi_categories.csv").set_index("word").category
nouns["cat"] = nouns.word.map(cdi)
nouns = nouns[nouns.cat.isin(CATS)]


def gaps(S, lab, cats, N):
    """-> {category: within-minus-between cohesion gap} for one labelling."""
    out = {}
    for c in cats:
        m = lab == c
        n = int(m.sum())
        within = (S[np.ix_(m, m)].sum() - n) / (n * (n - 1))
        between = S[np.ix_(m, ~m)].sum() / (n * (N - n))
        out[c] = within - between
    return out


rows = []
for run in runs:
    z = np.load(EMB / f"{run}.npz", allow_pickle=True)
    words = [str(w) for w in z["words"]]
    ix = {w: i for i, w in enumerate(words)}
    d = nouns[nouns.word.isin(ix)]
    W = z["W"].astype(np.float32)[[ix[w] for w in d.word]]
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    S = W @ W.T
    lab = np.asarray(d.cat.astype(str))
    N = len(lab)
    cats = [c for c in CATS if (lab == c).sum() >= MIN_N]

    obs = gaps(S, lab, cats, N)
    iu = np.triu_indices(N, 1)
    same = (lab[:, None] == lab[None, :])[iu]
    obs_all = S[iu][same].mean() - S[iu][~same].mean()

    rng = np.random.default_rng(a.seed)
    null = {c: np.empty(a.perms) for c in cats}
    null_all = np.empty(a.perms)
    for k in range(a.perms):
        p = rng.permutation(lab)
        for c, g in gaps(S, p, cats, N).items():
            null[c][k] = g
        sm = (p[:, None] == p[None, :])[iu]
        null_all[k] = S[iu][sm].mean() - S[iu][~sm].mean()

    seed = int(re.search(r"_s(\d+)$", run).group(1))
    for c in cats:
        m = lab == c
        n = int(m.sum())
        nu = null[c]
        rows.append(dict(
            family=a.family, seed=seed, category=c, n=n,
            within=float((S[np.ix_(m, m)].sum() - n) / (n * (n - 1))),
            between=float(S[np.ix_(m, ~m)].sum() / (n * (N - n))),
            gap=float(obs[c]),
            null_mean=float(nu.mean()), null_sd=float(nu.std(ddof=1)),
            null_lo=float(np.percentile(nu, 2.5)), null_hi=float(np.percentile(nu, 97.5)),
            p_perm=float((nu >= obs[c]).mean()), n_perm=a.perms))
    rows.append(dict(
        family=a.family, seed=seed, category="ALL", n=N,
        within=float(S[iu][same].mean()), between=float(S[iu][~same].mean()),
        gap=float(obs_all), null_mean=float(null_all.mean()),
        null_sd=float(null_all.std(ddof=1)),
        null_lo=float(np.percentile(null_all, 2.5)),
        null_hi=float(np.percentile(null_all, 97.5)),
        p_perm=float((null_all >= obs_all).mean()), n_perm=a.perms))
    print(f"  {run}: pooled gap {obs_all:.4f} (p={(null_all >= obs_all).mean():.4f})")

out = pd.DataFrame(rows)
dst = R / "results" / f"lexicon_category_structure_{a.family}.csv"
out.to_csv(dst, index=False)

# Holm across categories, on the seed-mean p (reported, not written)
g = (out[out.category != "ALL"].groupby("category")
        .agg(n=("n", "first"), gap=("gap", "mean"), sd=("gap", "std"),
             p=("p_perm", "max")).sort_values("p"))
k = len(g)
g["holm"] = [min(1.0, p * (k - i)) for i, p in enumerate(g.p)]
print(g.round(4).to_string())
print(f"\nwrote {dst.relative_to(R)}  ({out.seed.nunique()} seeds x {k} categories, "
      f"{a.perms} permutations each)")
