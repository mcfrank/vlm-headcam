"""Weighted bag-of-words: pool the utterance's word embeddings weighted by a per-word cue
(a static word->weight prior) instead of uniformly, so the referent word dominates the text
vector. Tests how much of the word-selection headroom (filtnat 68.5 -> t15 73.2) an accessible
per-word cue recovers. Reuses region-MIL; only the text pooling changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import save_json
from train import build_vocab, encode
from train_region_mil import RegionMIL, RegionPairs, eval_4afc_region, load_region_cache, collate


class WeightedRegionMIL(RegionMIL):
    def set_prior(self, vec):
        self.register_buffer("wprior", vec)

    def enc_text(self, t, n):
        e = self.word(t)                                  # [B,L,D]
        w = self.wprior[t] * (t != 0).float()             # [B,L] per-word weight, pad=0
        pooled = (e * w.unsqueeze(-1)).sum(1) / w.sum(1).clamp(min=1e-6).unsqueeze(-1)
        return F.normalize(pooled, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--region-cache", required=True)
    ap.add_argument("--eval-frames", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--word-prior", required=True, help="json word->weight; unknown words get --floor")
    ap.add_argument("--floor", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    man = pd.read_parquet(a.manifest)
    vocab = build_vocab(man.text, 5)
    prior = json.load(open(a.word_prior))
    pvec = torch.tensor([1.0 if w == "<pad>" else prior.get(w, a.floor) for w in vocab],
                        dtype=torch.float32).to(dev)

    emb, lut = load_region_cache(a.region_cache)
    ds = RegionPairs(emb, lut, man, vocab)
    print(f"pairs {len(ds)} | vocab {len(vocab)} | prior covers "
          f"{np.mean([w in prior for w in vocab])*100:.0f}% of vocab", flush=True)
    _cov = len(ds) / max(len(man), 1)
    if _cov < 0.999:   # a cache that does not span the manifest silently shrinks training
        print(f"  COVERAGE {100*_cov:.1f}% — {len(man)-len(ds)} of {len(man)} manifest "
              f"pairs have no cached frame", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     collate_fn=collate, num_workers=4)
    m = WeightedRegionMIL(len(vocab)).to(dev); m.set_prior(pvec)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1)
    ev = pd.read_parquet(a.eval_frames); ecache, elut = load_region_cache(a.eval_cache)

    best = 0.0
    for ep in range(a.epochs):
        m.train()
        for _, v, t, n in dl:
            v, t, n = v.to(dev), t.to(dev), n.to(dev)
            w = torch.ones(len(v), device=dev)
            loss = m.forward_loss(v, t, n, w)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = eval_4afc_region(m, ecache, elut, ev, vocab, dev)
        best = max(best, acc)
        print({"ep": ep, "acc": round(acc, 4)}, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), Path(a.out) / "model.pt")
    save_json(vocab, Path(a.out) / "vocab.json")
    print(f"DONE {a.out} best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
