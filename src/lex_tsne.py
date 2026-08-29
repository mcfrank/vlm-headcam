"""t-SNE of a learned lexicon (default: the best free model, B26_lad_base_s0).

Words kept: frequency >= --min-count occurrences in the transcript POS table (which also
supplies the majority POS for coloring). -> results/lexicon_tsne.csv
(word, x, y, pos, count, konkle60).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

R = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="B26_lad_base_s0")
ap.add_argument("--cache", default="._lexicon_cache")
ap.add_argument("--min-count", type=int, default=50)
ap.add_argument("--min-nn", type=float, default=4.0,
                help="keep words whose top-3-NN mean cosine exceeds this multiple of the "
                     "random-vector null (1/sqrt(dim)); 0 disables")
ap.add_argument("--perplexity", type=float, default=30)
ap.add_argument("--pos", default=None, help="comma list of POS to keep (e.g. NOUN)")
a = ap.parse_args()

z = np.load(Path(a.cache) / "emb" / f"{a.run}.npz", allow_pickle=True)
words = [str(w) for w in z["words"]]
W = z["W"].astype(np.float32)
W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8

pos = pd.read_csv(Path(a.cache) / "word_pos.csv").set_index("word")
keep = [i for i, w in enumerate(words)
        if w != "<pad>" and w in pos.index and pos.loc[w, "count"] >= a.min_count]
if a.pos:
    ok = set(a.pos.split(","))
    keep = [i for i in keep if pos.loc[words[i], "pos"] in ok]
S = W[keep] @ W[keep].T
np.fill_diagonal(S, -1)
nn_cos = np.sort(S, axis=1)[:, -3:].mean(1)
if a.min_nn:
    null = 1 / np.sqrt(W.shape[1])
    m = nn_cos > a.min_nn * null
    keep = [k for k, ok in zip(keep, m) if ok]; nn_cos = nn_cos[m]
print(f"{a.run}: {len(keep)} of {len(words)} words (count >= {a.min_count}, "
      f"top3-NN > {a.min_nn}x null)")

X = TSNE(n_components=2, perplexity=a.perplexity, metric="cosine", init="pca",
         random_state=0).fit_transform(W[keep])
K60 = set(pd.read_csv(R / "results" / "item_eval_b26.csv").query("set=='test60'").category)
out = pd.DataFrame(dict(word=[words[i] for i in keep], x=X[:, 0], y=X[:, 1],
                        pos=[pos.loc[words[i], "pos"] for i in keep],
                        count=[pos.loc[words[i], "count"] for i in keep], nn_cos=nn_cos))
out["konkle60"] = out.word.isin(K60)
tag = f"_{a.pos.replace(',','-')}" if a.pos else ""
out.to_csv(R / "results" / f"lexicon_tsne_{a.run}{tag}.csv", index=False)
print(f"wrote results/lexicon_tsne_{a.run}{tag}.csv")
