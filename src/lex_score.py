"""Score every extracted lexicon (two-tower runs and word2vec baselines) against the human
word-relatedness sets in data/word_sim (bilingual-babyLM protocol: pairs with both words in
vocab), and compute model-vs-word2vec RSA at matched scale.

-> results/lexicon_relatedness.csv  one row per (source, dataset): spearman, n_pairs, coverage
-> results/lexicon_rsa.csv          per scale: RSA (spearman over shared-vocab pairwise cosines)
"""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="._lexicon_cache")
a = ap.parse_args()
cache = Path(a.cache)

# per-file parsing: (sep, colnames) — Bellezza is single-word norms, not pairs (excluded)
FMT = {"wordsim353": (";", None), "simlex999": (";", None), "rg65": (";", None),
       "MTest-3000": (r"\s+", None), "SimVerb-3500": ("\t", "v")}
DATASETS = {k: R / "data" / "word_sim" / f"{k}.csv" for k in FMT}


def read_pairs(name):
    sep, kind = FMT[name]
    if kind == "v":
        d = pd.read_csv(DATASETS[name], sep=sep, header=None,
                        names=["w1", "w2", "pos", "score", "rel"], engine="python")
    else:
        d = pd.read_csv(DATASETS[name], sep=sep, header=None, names=["w1", "w2", "score"],
                        engine="python")
    d["w1"] = d.w1.astype(str).str.strip().str.lower()
    d["w2"] = d.w2.astype(str).str.strip().str.lower()
    d["score"] = pd.to_numeric(d.score, errors="coerce")
    return d.dropna(subset=["score"])


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    words = [str(w) for w in z["words"]]
    W = z["W"].astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    return {w: i for i, w in enumerate(words)}, W


def relatedness(vocab, W):
    out = []
    for name in DATASETS:
        d = read_pairs(name)
        d = d[d.w1 != d.w2]
        ok = d[d.w1.isin(vocab) & d.w2.isin(vocab)]
        if len(ok) < 20:
            out.append((name, np.nan, len(ok), len(d))); continue
        sims = [W[vocab[r.w1]] @ W[vocab[r.w2]] for r in ok.itertuples()]
        out.append((name, spearmanr(sims, ok.score).correlation, len(ok), len(d)))
    return out


rows = []
sources = sorted((cache / "emb").glob("*.npz")) + sorted((cache / "w2v").glob("*.npz"))
for p in sources:
    kind = "w2v" if p.parent.name == "w2v" else "model"
    vocab, W = load_npz(p)
    for name, rho, n_ok, n_all in relatedness(vocab, W):
        rows.append(dict(source=p.stem, kind=kind, dataset=name, spearman=rho,
                         n_pairs=n_ok, n_total=n_all, vocab=len(vocab)))
pd.DataFrame(rows).to_csv(R / "results" / "lexicon_relatedness.csv", index=False)
print("relatedness: ", len(rows), "rows")

# ---- RSA: two-tower vs word2vec on the same draw ---------------------------------
TOPN = 2000                    # most frequent shared words (vocabs are frequency-ordered)
rsa = []
for wp in sorted((cache / "w2v").glob("bv26_rand_*_s0.npz")):
    n = int(re.search(r"rand_(\d+)_", wp.name).group(1))
    for mp in sorted((cache / "emb").glob(f"B26_rand_{n}_s*.npz")):
        v1, W1 = load_npz(mp); v2, W2 = load_npz(wp)
        shared = [w for w in v1 if w in v2 and w != "<pad>"][:TOPN]
        if len(shared) < 100:
            continue
        A = W1[[v1[w] for w in shared]]; B = W2[[v2[w] for w in shared]]
        iu = np.triu_indices(len(shared), 1)
        rho = spearmanr((A @ A.T)[iu], (B @ B.T)[iu]).correlation
        rsa.append(dict(n_pairs_scale=n, run=mp.stem, w2v=wp.stem, n_shared=len(shared), rsa=rho))
# and the full corpus
for mp in sorted((cache / "emb").glob("B26_lad_base_s*.npz")):
    wp = cache / "w2v" / "bv26_base.npz"
    if wp.exists():
        v1, W1 = load_npz(mp); v2, W2 = load_npz(wp)
        shared = [w for w in v1 if w in v2 and w != "<pad>"][:TOPN]
        A = W1[[v1[w] for w in shared]]; B = W2[[v2[w] for w in shared]]
        iu = np.triu_indices(len(shared), 1)
        rsa.append(dict(n_pairs_scale=1815243, run=mp.stem, w2v=wp.stem,
                        n_shared=len(shared), rsa=spearmanr((A @ A.T)[iu], (B @ B.T)[iu]).correlation))
pd.DataFrame(rsa).to_csv(R / "results" / "lexicon_rsa.csv", index=False)
print("rsa:", len(rsa), "rows")
