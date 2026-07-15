"""Exp 3 (chapter: architectures linking language & vision): a tiny FROM-SCRATCH captioner.
Frozen DINOv2 region features (emb_reg [N,17,768]) are the cross-attention memory; a small
autoregressive Transformer decoder (no pretrained LLM) generates the utterance token-by-token.
Alignment lives in the decoder cross-attention (words query patches). Trained by next-token CE.
Konkle 4AFC is scored generatively: pick the image under which the target word is most likely
(continuation from BOS) -- argmax_i P(word | image_i)."""
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


class Captioner(nn.Module):
    def __init__(self, vocab, D, d=384, layers=3, heads=6, drop=0.1, max_len=24):
        super().__init__()
        self.bos, self.max_len = vocab, max_len          # BOS id = vocab; EOS = vocab+1
        V = vocab + 2
        self.patch_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.tok = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.head = nn.Linear(d, V)

    def decode(self, mem, seq):                          # mem [B,R,d], seq [B,T] -> logits [B,T,V]
        T = seq.shape[1]
        x = self.tok(seq) + self.pos(torch.arange(T, device=seq.device))[None]
        cm = torch.triu(torch.full((T, T), float("-inf"), device=seq.device), 1)
        out = self.dec(x, mem, tgt_mask=cm, tgt_key_padding_mask=(seq == 0))
        return self.head(out)

    def forward(self, patches, seq):
        return self.decode(self.patch_proj(patches), seq)


class Caps(torch.utils.data.Dataset):
    def __init__(self, emb, lut, man, vocab, bos, max_len=24):
        self.emb, self.bos, self.eos, self.max_len = emb, bos, bos + 1, max_len
        self.rows = []
        for v, f, t in zip(man.video_id, man.frame_idx, man.text):
            k = frame_key(v, int(f))
            if k in lut:
                ids = encode(t, vocab, max_len - 2)
                if ids:
                    self.rows.append((lut[k], ids))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, ids = self.rows[i]
        seq = [self.bos] + ids + [self.eos]
        s = torch.zeros(self.max_len, dtype=torch.long); s[:len(seq)] = torch.tensor(seq)
        v = torch.from_numpy(np.asarray(self.emb[r], dtype=np.float32))
        return v, s


def collate(b):
    return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])


@torch.no_grad()
def eval_4afc(m, eemb, elut, ev, vocab, dev, n_trials=100, seed=0):
    rng = np.random.default_rng(seed)
    pools, cat_tok = {}, {}
    for cat, g in ev.groupby("category"):
        ids = encode(cat, vocab, 16)
        rows = [elut[frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx) if frame_key(v, f) in elut]
        if ids and len(rows) >= 4:
            pools[cat] = rows; cat_tok[cat] = ids[0]         # score first token of the category word
    cats = sorted(pools)
    allr = sorted({r for rs in pools.values() for r in rs})
    mem = m.patch_proj(torch.from_numpy(np.asarray(eemb[allr], dtype=np.float32)).to(dev))
    r2i = {r: i for i, r in enumerate(allr)}
    bos = torch.full((1, 1), m.bos, dtype=torch.long, device=dev)
    accs = []
    for cat in cats:
        c = cat_tok[cat]; others = [x for x in cats if x != cat]; correct = 0
        for _ in range(n_trials):
            cand = [rng.choice(pools[cat])] + [rng.choice(pools[x]) for x in rng.choice(others, 3, replace=False)]
            mm = mem[[r2i[r] for r in cand]]                 # [4,R,d]
            logits = m.decode(mm, bos.expand(4, 1))[:, 0]     # [4,V] first-step
            p_c = torch.log_softmax(logits, -1)[:, c]         # logP(word | image_i)
            correct += int(p_c.argmax().item() == 0)
        accs.append(correct / n_trials)
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-cache", default=f"{W}/emb_reg")
    ap.add_argument("--eval-cache", default=f"{W}/emb_konkle")
    ap.add_argument("--manifest", default=f"{W}/manifests/grid_baseline_train.parquet")
    ap.add_argument("--eval-frames", default=f"{W}/manifests/eval_frames_konkle.parquet")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=6)
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
    global vocab
    vocab = build_vocab(man.text, 5)
    ds = Caps(emb, lut, man, vocab, len(vocab))
    D = np.asarray(ds[0][0]).shape[-1]
    print(f"pairs {len(ds)} | vocab {len(vocab)} | D {D} | dec {a.layers}x{a.dim}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     collate_fn=collate, num_workers=4)
    m = Captioner(len(vocab), D, a.dim, a.layers, a.heads).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1)
    best = 0.0
    for ep in range(a.epochs):
        m.train(); tot = 0.0
        for v, s in dl:
            v, s = v.to(dev), s.to(dev)
            logits = m(v, s[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), s[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        m.eval()
        acc = eval_4afc(m, eemb, elut, ev, vocab, dev, seed=a.seed)
        best = max(best, acc)
        print({"ep": ep, "loss": round(tot / len(dl), 3), "acc": round(acc, 4)}, flush=True)
    json.dump(vocab, open(out / "vocab.json", "w"))
    torch.save(m.state_dict(), out / "model.pt")
    print(f"DONE {a.out} best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
