"""Extract the learned lexicons and the material for the language-only baseline (runs on ccn2).

For every B26/C8 run: the text tower's embedding matrix (model.word.weight) + its vocab, saved
as scratch/lexicon/emb/<run>.npz {words, W}. The bag-of-words tower means each row IS the
word's representation.

Also exports, for the word2vec baseline and the t-SNE coloring:
  scratch/lexicon/text/<manifest>.txt.gz   one utterance per line, tokens space-joined with the
                                           SAME tokenizer the model saw (common.tokenize)
  scratch/lexicon/word_pos.csv             majority spacy POS + count per token, from the
                                           2025.2 merged transcript parse
"""
import gzip, json, re, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from common import tokenize

ROOT = Path("/data2/mcfrank/vlm-headcam")
OUT = ROOT / "scratch" / "lexicon"
(OUT / "emb").mkdir(parents=True, exist_ok=True)
(OUT / "text").mkdir(exist_ok=True)

# F_* = the FINAL audio-filtered corpus (bv26a manifests); B26_/C8_ are preview corpora
# kept so the earlier figures still rebuild.
RUN_RE = re.compile(r"^(B26_(rand_\d+|lad\d*_?(base|filtnat|t15|t2))"
                    r"|C8_dinov3l(_bv)?_grid4x4_(rand_\d+|base)"
                    r"|F_(dinov3b|dinov3l|vitb_bv|vits_bv)_(rand_\d+|base))_s\d$")

# ---- 1. embedding matrices -------------------------------------------------------
done = 0
for rd in sorted(ROOT.glob("runs/*")):
    if not RUN_RE.match(rd.name) or not (rd / "model.pt").exists():
        continue
    dst = OUT / "emb" / f"{rd.name}.npz"
    if dst.exists():
        continue
    vocab = json.loads((rd / "vocab.json").read_text())
    words = [w for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    sd = torch.load(rd / "model.pt", map_location="cpu")
    W = sd["word.weight"].numpy().astype(np.float16)
    assert W.shape[0] == len(words), rd.name
    np.savez_compressed(dst, words=np.array(words), W=W)
    done += 1
print(f"embeddings: wrote {done} new npz")

# ---- 2. tokenized utterance text per manifest ------------------------------------
mans = (sorted(ROOT.glob("manifests/bv26_rand_*_s0.parquet"))
        + [ROOT / "manifests" / "bv26_base.parquet"]
        + sorted(ROOT.glob("manifests/bv26a_rand_*_s0.parquet"))       # FINAL corpus
        + [ROOT / "manifests" / "bv26a_base.parquet"])
for mp in mans:
    dst = OUT / "text" / (mp.stem + ".txt.gz")
    if dst.exists():
        continue
    texts = pd.read_parquet(mp, columns=["text"]).text
    with gzip.open(dst, "wt") as f:
        for t in texts:
            f.write(" ".join(tokenize(t)) + "\n")
    print("text:", dst.name, len(texts))

# ---- 3. majority POS per token ---------------------------------------------------
pos_dst = OUT / "word_pos.csv"
if not pos_dst.exists():
    T = "/ccn2a/dataset/babyview/2025.2/outputs/merged_transcripts_parsed.csv"
    counts = defaultdict(Counter)
    for ch in pd.read_csv(T, usecols=["spacy_token_text", "spacy_pos"], chunksize=2_000_000,
                          low_memory=False):
        ch = ch.dropna()
        for tok, pos in zip(ch.spacy_token_text.astype(str), ch.spacy_pos.astype(str)):
            for t in tokenize(tok):
                counts[t][pos] += 1
    rows = [(w, c.most_common(1)[0][0], sum(c.values())) for w, c in counts.items()]
    pd.DataFrame(rows, columns=["word", "pos", "count"]).to_csv(pos_dst, index=False)
    print("pos: wrote", pos_dst.name, len(rows))
print("ALL DONE")
