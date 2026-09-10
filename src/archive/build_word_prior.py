"""Build a global per-word 'contentness' prior from the token transcript: content nouns get
weight 1.0, everything else a low floor. This is the noun-bias word-selection cue (static, no
audio, no per-pair alignment) for the weighted bag-of-words encoder."""
import json
import pandas as pd
from common import PARSED

tok = pd.read_csv(PARSED, usecols=["token", "spacy_pos", "spacy_is_stop"])
tok = tok.dropna(subset=["token"])
tok["w"] = tok.token.str.lower().str.strip()
tok = tok[tok.w.str.match(r"^[a-z]+$", na=False)]

# per word: dominant POS + stop flag
agg = tok.groupby("w").agg(pos=("spacy_pos", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
                           stop=("spacy_is_stop", lambda s: s.mean() > 0.5),
                           n=("w", "size")).reset_index()
agg = agg[agg.n >= 3]
content = agg.pos.isin(["NOUN", "PROPN"]) & (~agg.stop)
prior = {w: (1.0 if c else 0.15) for w, c in zip(agg.w, content)}
json.dump(prior, open("manifests/word_prior_noun.json", "w"))
print(f"word prior: {len(prior)} words | content nouns {int(content.sum())} "
      f"({content.mean()*100:.0f}%) | e.g. content: {[w for w,c in zip(agg.w,content) if c][:8]}")
