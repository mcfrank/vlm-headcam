"""CDI-category structure in the learned noun lexicon: mean within- vs between-category
cosine per category (in the full embedding space), plus a label-permutation null for the
overall within-between gap. -> results/lexicon_category_structure.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
RUN = "B26_lad_base_s0"
CATS = ["animals", "food_drink", "vehicles", "toys", "clothing", "body_parts",
        "household", "furniture_rooms", "outside", "places"]

z = np.load(R / "._lexicon_cache" / "emb" / f"{RUN}.npz", allow_pickle=True)
words = [str(w) for w in z["words"]]
W = z["W"].astype(np.float32)
W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
ix = {w: i for i, w in enumerate(words)}

d = pd.read_csv(R / "results" / f"lexicon_tsne_{RUN}_NOUN.csv")
cdi = pd.read_csv(R / "results" / "cdi_categories.csv").set_index("word").category
d["cat"] = d.word.map(cdi)
d = d[d.cat.isin(CATS) & d.word.isin(ix)]
E = W[[ix[w] for w in d.word]]
S = E @ E.T
lab = np.asarray(d.cat.astype(str))
iu = np.triu_indices(len(d), 1)
same = (lab[:, None] == lab[None, :])[iu]
obs = S[iu][same].mean() - S[iu][~same].mean()
rng = np.random.default_rng(0)
null = []
for _ in range(5000):
    p = rng.permutation(lab)
    sm = (p[:, None] == p[None, :])[iu]
    null.append(S[iu][sm].mean() - S[iu][~sm].mean())
null = np.array(null)

rows = []
for c in CATS:
    m = lab == c
    if m.sum() < 4:
        continue
    rows.append(dict(category=c, n=int(m.sum()),
                     within=float(S[np.ix_(m, m)][np.triu_indices(m.sum(), 1)].mean()),
                     between=float(S[np.ix_(m, ~m)].mean())))
out = pd.DataFrame(rows)
out["gap_overall"] = obs
out["null_mean"] = null.mean(); out["null_sd"] = null.std()
out["p_perm"] = float((null >= obs).mean())
out.to_csv(R / "results" / "lexicon_category_structure.csv", index=False)
print(out.round(4).to_string(index=False))
