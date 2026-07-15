"""Architecture experiments (chapter: architectures linking language & vision).
Frozen DINOv2 region features (emb_reg [N,17,768]); vary ONLY how a word aligns to regions:
  --pool max        MIL max over regions (our baseline)
  --pool soft       softmax(sim/tau)-weighted pool over regions (Exp 2a; tau->0 == max)
  --pool attn       learned cross-attention pooling W_q/W_k/W_v over region values (Exp 2b)
Everything else (frozen features, bag-of-words text, InfoNCE, Konkle 4AFC) is held fixed."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from train import build_vocab, encode
from train_region_mil import load_region_cache
from common import frame_key

W = "/data2/mcfrank/vlm-headcam"


class ArchMIL(nn.Module):
    def __init__(self, vocab, D, d=512, pool="max", tau=0.05, drop=0.1):
        super().__init__()
        self.vproj = nn.Sequential(nn.LayerNorm(D), nn.Dropout(drop), nn.Linear(D, d))
        self.word = nn.Embedding(vocab, d, padding_idx=0)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))
        self.pool, self.tau, self.d = pool, tau, d
        if pool == "attn":
            self.Wq = nn.Linear(d, d, bias=False)
            self.Wk = nn.Linear(d, d, bias=False)
            self.Wv = nn.Linear(d, d, bias=False)

    def enc_regions(self, v):                       # [.,R,D] -> [.,R,d] normalized
        return F.normalize(self.vproj(v), dim=-1)

    def enc_text(self, t, n):                        # bag-of-words -> [.,d] normalized
        e = self.word(t); m = (t != 0).unsqueeze(-1).float()
        return F.normalize((e * m).sum(1) / n.clamp(min=1).unsqueeze(-1).float(), dim=-1)

    def score(self, R, T):                           # R [B,Rn,d], T [M,d] -> S [B,M]
        e = torch.einsum('brd,md->bmr', R, T)        # per-region similarity
        if self.pool == "max":
            return e.max(-1).values
        if self.pool == "soft":
            a = torch.softmax(e / self.tau, -1)
            return (a * e).sum(-1)
        q = self.Wq(T); k = self.Wk(R); vv = self.Wv(R)          # attn
        a = torch.softmax(torch.einsum('brd,md->bmr', k, q) / self.d ** 0.5, -1)
        z = torch.einsum('bmr,brd->bmd', a, vv)
        return torch.einsum('bmd,md->bm', z, T)

    def loss(self, R, T):
        S = self.logit_scale.clamp(max=np.log(100)).exp() * self.score(R, T)   # [B,B]
        lab = torch.arange(len(R), device=R.device)
        return 0.5 * (F.cross_entropy(S, lab) + F.cross_entropy(S.t(), lab))


class Pairs(torch.utils.data.Dataset):
    def __init__(self, emb, lut, man, vocab, max_len=16):
        self.emb, self.max_len = emb, max_len
        self.rows = []
        for v, f, t in zip(man.video_id, man.frame_idx, man.text):
            k = frame_key(v, int(f))
            if k in lut:
                ids = encode(t, vocab, max_len)
                if ids:
                    self.rows.append((lut[k], ids))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, ids = self.rows[i]
        v = torch.from_numpy(np.asarray(self.emb[r], dtype=np.float32))
        t = torch.zeros(self.max_len, dtype=torch.long); t[:len(ids)] = torch.tensor(ids)
        return v, t, len(ids)


def collate(b):
    return (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]),
            torch.tensor([x[2] for x in b]))


@torch.no_grad()
def eval_4afc(m, emb, lut, ev, vocab, dev, n_trials=100, seed=0, max_len=16):
    rng = np.random.default_rng(seed)
    pools, cat_ids = {}, {}
    for cat, g in ev.groupby("category"):
        ids = encode(cat, vocab, max_len)
        rows = [lut[frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx) if frame_key(v, f) in lut]
        if ids and len(rows) >= 4:
            pools[cat] = rows; cat_ids[cat] = ids
    cats = sorted(pools)
    allr = sorted({r for rs in pools.values() for r in rs})
    R = m.enc_regions(torch.from_numpy(np.asarray(emb[allr], dtype=np.float32)).to(dev))
    r2i = {r: i for i, r in enumerate(allr)}
    accs = []
    for cat in cats:
        toks = cat_ids[cat]
        t = torch.zeros(1, max_len, dtype=torch.long, device=dev); t[0, :len(toks)] = torch.tensor(toks, device=dev)
        tv = m.enc_text(t, torch.tensor([len(toks)], device=dev))
        others = [c for c in cats if c != cat]; correct = 0
        for _ in range(n_trials):
            cand = [rng.choice(pools[cat])] + [rng.choice(pools[c]) for c in rng.choice(others, 3, replace=False)]
            sc = m.score(R[[r2i[r] for r in cand]], tv).squeeze(-1)     # [4]
            correct += int(sc.argmax().item() == 0)
        accs.append(correct / n_trials)
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-cache", default=f"{W}/emb_reg")
    ap.add_argument("--eval-cache", default=f"{W}/emb_konkle")
    ap.add_argument("--manifest", default=f"{W}/manifests/grid_baseline_train.parquet")
    ap.add_argument("--eval-frames", default=f"{W}/manifests/eval_frames_konkle.parquet")
    ap.add_argument("--pool", choices=["max", "soft", "attn"], default="max")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    emb, lut = load_region_cache(a.region_cache)
    eemb, elut = load_region_cache(a.eval_cache)
    man = pd.read_parquet(a.manifest); ev = pd.read_parquet(a.eval_frames)
    vocab = build_vocab(man.text, 5)
    ds = Pairs(emb, lut, man, vocab)
    D = np.asarray(ds[0][0]).shape[-1]
    print(f"pairs {len(ds)} | vocab {len(vocab)} | pool {a.pool} tau {a.tau} | D {D}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     collate_fn=collate, num_workers=4)
    m = ArchMIL(len(vocab), D, a.dim, a.pool, a.tau).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1)
    best = 0.0
    for ep in range(a.epochs):
        m.train()
        for v, t, n in dl:
            v, t, n = v.to(dev), t.to(dev), n.to(dev)
            R = m.enc_regions(v); T = m.enc_text(t, n)
            loss = m.loss(R, T)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        acc = eval_4afc(m, eemb, elut, ev, vocab, dev, seed=a.seed)
        best = max(best, acc)
        print({"ep": ep, "acc": round(acc, 4)}, flush=True)
    json.dump(vocab, open(out / "vocab.json", "w"))
    torch.save(m.state_dict(), out / "model.pt")
    print(f"DONE {a.out} best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
