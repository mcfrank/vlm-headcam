"""Per-pair, per-word weighted bag-of-words. Reads a cue manifest (words + a per-word weight
column) and pools the text embeddings weighted by that cue, so a context-dependent cue (discourse
newness, prosody) can pick the referent among several content words. Uniform (--weight-col none)
is the matched control on the same text source."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import frame_key, save_json, tokenize
from train import build_vocab
from train_region_mil import RegionMIL, eval_4afc_region, load_region_cache


def build_vocab_words(man, min_freq=5):
    from collections import Counter
    c = Counter(tok for ws in man.words for tok in tokenize(str(ws)))
    vocab = {"<pad>": 0}
    for w, n in c.most_common():
        if n >= min_freq:
            vocab[w] = len(vocab)
    return vocab


class WeightedPairs(torch.utils.data.Dataset):
    def __init__(self, emb, lut, man, vocab, weight_col, max_len=16):
        uniform = weight_col is None or weight_col not in man.columns
        self.rows, self.ids, self.wts = [], [], []
        for r in man.itertuples(index=False):
            k = frame_key(r.video_id, r.frame_idx)
            if k not in lut:
                continue
            words = str(r.words).split()
            wts = [1.0] * len(words) if uniform else [float(x) for x in str(getattr(r, weight_col)).split()]
            if len(wts) != len(words):
                wts = [1.0] * len(words)
            ids, w = [], []
            for word, wt in zip(words, wts):
                for tk in tokenize(word):
                    if tk in vocab:
                        ids.append(vocab[tk]); w.append(wt)
            if not ids:
                continue
            self.rows.append(lut[k]); self.ids.append(ids[:max_len]); self.wts.append(w[:max_len])
        self.emb, self.max_len = emb, max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        v = torch.from_numpy(np.asarray(self.emb[self.rows[i]], dtype=np.float32))
        t = torch.zeros(self.max_len, dtype=torch.long); wt = torch.zeros(self.max_len)
        L = len(self.ids[i]); t[:L] = torch.tensor(self.ids[i]); wt[:L] = torch.tensor(self.wts[i])
        return i, v, t, L, wt


def collate(b):
    return (torch.tensor([x[0] for x in b]), torch.stack([x[1] for x in b]),
            torch.stack([x[2] for x in b]), torch.tensor([x[3] for x in b]),
            torch.stack([x[4] for x in b]))


class PerWordMIL(RegionMIL):
    def enc_text_w(self, t, wt):
        e = self.word(t)
        w = wt * (t != 0).float()
        pooled = (e * w.unsqueeze(-1)).sum(1) / w.sum(1).clamp(min=1e-6).unsqueeze(-1)
        return F.normalize(pooled, dim=-1)

    def loss_w(self, v, t, wt):
        R = self.enc_regions(v); T = self.enc_text_w(t, wt)
        logits = self.logit_scale.clamp(max=np.log(100)).exp() * self.sims(R, T)
        lab = torch.arange(len(v), device=v.device)
        return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--region-cache", required=True)
    ap.add_argument("--eval-frames", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--weight-col", default=None, help="per-word weight column; omit for uniform control")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    man = pd.read_parquet(a.manifest)
    vocab = build_vocab_words(man)
    emb, lut = load_region_cache(a.region_cache)
    ds = WeightedPairs(emb, lut, man, vocab, a.weight_col)
    print(f"pairs {len(ds)} | vocab {len(vocab)} | weight_col {a.weight_col or 'UNIFORM'}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     collate_fn=collate, num_workers=4)
    m = PerWordMIL(len(vocab)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1)
    ev = pd.read_parquet(a.eval_frames); ecache, elut = load_region_cache(a.eval_cache)

    best = 0.0
    for ep in range(a.epochs):
        m.train()
        for _, v, t, n, wt in dl:
            v, t, wt = v.to(dev), t.to(dev), wt.to(dev)
            loss = m.loss_w(v, t, wt)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = eval_4afc_region(m, ecache, elut, ev, vocab, dev)
        best = max(best, acc)
        print({"ep": ep, "acc": round(acc, 4)}, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), Path(a.out) / "model.pt"); save_json(vocab, Path(a.out) / "vocab.json")
    print(f"DONE {a.out} best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
