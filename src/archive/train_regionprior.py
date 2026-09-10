"""Per-pair region prior: bias region-MIL's cell selection toward a pose-indicated target cell
(the caregiver's hand/face). Select-with-prior, score-raw-sim (like --center-prior but per pair).
Tests whether steering attention to the pointed/held object helps referent identification."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import frame_key, save_json
from train import build_vocab, encode
from train_region_mil import RegionMIL, eval_4afc_region, load_region_cache


class RegionPriorPairs(torch.utils.data.Dataset):
    def __init__(self, emb, lut, man, vocab, strength, max_len=16):
        self.rows, self.ids, self.tc = [], [], []
        for r in man.itertuples(index=False):
            k = frame_key(r.video_id, r.frame_idx)
            if k not in lut:
                continue
            toks = encode(r.text, vocab, max_len)
            if not toks:
                continue
            self.rows.append(lut[k]); self.ids.append(toks); self.tc.append(int(r.target_cell))
        self.emb, self.max_len, self.ps = emb, max_len, strength
        self.Rn = int(np.asarray(emb[self.rows[0]]).shape[0])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        v = torch.from_numpy(np.asarray(self.emb[self.rows[i]], dtype=np.float32))
        t = torch.zeros(self.max_len, dtype=torch.long); L = len(self.ids[i]); t[:L] = torch.tensor(self.ids[i])
        prior = torch.zeros(self.Rn)
        if self.tc[i] >= 0:
            prior[self.tc[i]] = self.ps
        return i, v, t, L, prior


def collate(b):
    return (torch.tensor([x[0] for x in b]), torch.stack([x[1] for x in b]),
            torch.stack([x[2] for x in b]), torch.tensor([x[3] for x in b]),
            torch.stack([x[4] for x in b]))


class PriorMIL(RegionMIL):
    def loss_p(self, v, t, n, prior):
        R = self.enc_regions(v); T = self.enc_text(t, n)
        s = torch.einsum('brd,md->brm', R, T)               # [B, Rn, M]
        sel = (s + prior.unsqueeze(-1)).argmax(1, keepdim=True)  # bias which cell wins, per image
        scores = s.gather(1, sel).squeeze(1)                # [B, M] raw sim at selected cell
        logits = self.logit_scale.clamp(max=np.log(100)).exp() * scores
        lab = torch.arange(len(v), device=v.device)
        return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--region-cache", required=True)
    ap.add_argument("--eval-frames", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--strength", type=float, default=0.0, help="prior bonus (0 = uniform control)")
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
    emb, lut = load_region_cache(a.region_cache)
    ds = RegionPriorPairs(emb, lut, man, vocab, a.strength)
    cov = np.mean([c >= 0 for c in ds.tc])
    print(f"pairs {len(ds)} | vocab {len(vocab)} | strength {a.strength} | target-cell coverage {cov*100:.0f}%", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     collate_fn=collate, num_workers=4)
    m = PriorMIL(len(vocab)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1)
    ev = pd.read_parquet(a.eval_frames); ecache, elut = load_region_cache(a.eval_cache)

    best = 0.0
    for ep in range(a.epochs):
        m.train()
        for _, v, t, n, prior in dl:
            v, t, n, prior = v.to(dev), t.to(dev), n.to(dev), prior.to(dev)
            loss = m.loss_p(v, t, n, prior)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = eval_4afc_region(m, ecache, elut, ev, vocab, dev)
        best = max(best, acc)
        print({"ep": ep, "acc": round(acc, 4)}, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), Path(a.out) / "model.pt"); save_json(vocab, Path(a.out) / "vocab.json")
    print(f"DONE {a.out} best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
