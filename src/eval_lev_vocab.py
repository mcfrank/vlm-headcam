"""LEVANTE-vocab secondary eval (DevBench-style contrastive 4AFC). For each item, encode the
target word with the model's text tower and score it against each of the 4 candidate images
(max over regions); argmax = the model's choice (c0 is the correct image). Items whose target
word is out-of-vocabulary are marked unplayable. Writes per-item results joined to the child
IRT difficulty d, for a model-accuracy-vs-child-difficulty comparison."""
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from train_region_mil import RegionMIL, load_region_cache
from train import encode
from common import frame_key

W = "/data2/mcfrank/vlm-headcam"
run = sys.argv[1] if len(sys.argv) > 1 else "G_base_mil_full_s0"
dev = "cuda" if torch.cuda.is_available() else "cpu"

vocab = json.load(open(f"{W}/runs/{run}/vocab.json"))
emb, lut = load_region_cache(f"{W}/emb_lev_vocab")
m = RegionMIL(len(vocab), 512).to(dev)
m.load_state_dict(torch.load(f"{W}/runs/{run}/model.pt", map_location=dev)); m.eval()
items = pd.read_csv(f"{W}/lev_vocab_items.csv")


def img_regions(fname):
    key = frame_key(os.path.splitext(fname)[0], 0)
    if key not in lut:
        return None
    v = torch.from_numpy(np.asarray(emb[lut[key]], dtype=np.float32)).unsqueeze(0).to(dev)
    return m.enc_regions(v)[0]                                   # [R, D]


rows = []
with torch.no_grad():
    for r in items.itertuples(index=False):
        toks = encode(r.target_word, vocab, 16)
        in_vocab = bool(toks)
        rec = {"item_uid": r.item_uid, "target_word": r.target_word,
               "in_vocab": in_vocab, "d": r.d, "correct": np.nan, "p_correct": np.nan}
        cand = [img_regions(getattr(r, c)) for c in ("c0", "c1", "c2", "c3")]
        if in_vocab and all(c is not None for c in cand):
            t = torch.zeros(1, 16, dtype=torch.long, device=dev)
            t[0, :len(toks)] = torch.tensor(toks, device=dev)
            tv = m.enc_text(t, torch.tensor([len(toks)], device=dev))[0]         # [D]
            sc = torch.stack([torch.einsum("rd,d->r", c, tv).max() for c in cand])  # [4] cosine
            rec["correct"] = int(sc.argmax().item() == 0)
            # graded confidence uses the model's learned temperature (as in training logits)
            scale = m.logit_scale.clamp(max=np.log(100)).exp()
            rec["p_correct"] = torch.softmax(scale * sc, 0)[0].item()
        rows.append(rec)

res = pd.DataFrame(rows)
res.to_parquet(f"{W}/book_figs/lev_vocab_{run}.parquet", index=False)
played = res[res.correct.notna()]
withd = played[played.d.notna()]
print(f"items {len(res)} | in-vocab & playable {len(played)} | with IRT d {len(withd)}")
print(f"model 4AFC on playable items: {100*played.correct.mean():.1f}%  (chance 25)")
if len(withd) >= 8:
    # item bank d: higher = harder (verify empirically). easiness = -d.
    rho = withd[["correct", "d"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman(model correct, d) = {rho:+.3f}  over {len(withd)} items "
          f"(negative => model gets the child-EASY items right)")
