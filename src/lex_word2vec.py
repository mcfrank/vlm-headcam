"""Language-only baseline: word2vec (skip-gram) trained on the SAME utterance draws the
two-tower saw, exported by src/lex_extract.py (tokenized with the model's own tokenizer).

One model per text file. Hyperparameters: SGNS, dim 300, window 5, min_count 5 (the model
vocab's min-count), 10 epochs, fixed seed, 1 worker (determinism). Vectors ->
<cache>/w2v/<manifest>.npz {words, W}.
"""
import argparse, gzip, sys
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="._lexicon_cache")
a = ap.parse_args()
cache = Path(a.cache)
(cache / "w2v").mkdir(parents=True, exist_ok=True)

for tf in sorted((cache / "text").glob("*.txt.gz")):
    dst = cache / "w2v" / (tf.name.replace(".txt.gz", "") + ".npz")
    if dst.exists():
        continue
    with gzip.open(tf, "rt") as f:
        sents = [line.split() for line in f]
    m = Word2Vec(sents, vector_size=300, window=5, min_count=5, sg=1, negative=10,
                 epochs=10, seed=0, workers=1)
    words = m.wv.index_to_key
    np.savez_compressed(dst, words=np.array(words), W=m.wv.vectors.astype(np.float16))
    print(f"{dst.name}: {len(sents):,} utterances -> {len(words):,} words")
