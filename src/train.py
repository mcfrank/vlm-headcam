"""Two-tower contrastive training (CVCL-style) on frozen DINOv2 frame embeddings.

Vision tower : frozen DINOv2 emb (768) -> Linear proj -> dim (L2-normed)
Text tower   : word embeddings -> mean-pool over tokens -> dim (L2-normed)  [CVCL bag-of-words]
Loss         : symmetric InfoNCE, learnable temperature.

Reports per epoch: train loss, held-out val contrastive loss (does it diverge?), and
4AFC CDI-category accuracy (chance = 25%) on the labeled eval frames.
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from pathlib import Path

from common import MANIFEST_DIR, EMB_DIR, load_emb_cache, tokenize, EMB_DIM, save_json, frame_key


# ---------------- data ----------------
def build_vocab(texts, min_freq=5):
    c = Counter(t for txt in texts for t in tokenize(txt))
    vocab = {"<pad>": 0}
    for tok, n in c.most_common():
        if n >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(text, vocab, max_len=16):
    ids = [vocab[t] for t in tokenize(text) if t in vocab][:max_len]
    return ids


class Pairs(torch.utils.data.Dataset):
    def __init__(self, emb, lut, manifest, vocab, max_len=16):
        rows, self.ids, self.embrows = [], [], []
        has_key = "key" in manifest.columns
        for r in manifest.itertuples(index=False):
            key = r.key if has_key else frame_key(r.video_id, r.frame_idx)
            if key not in lut:
                continue
            ids = encode(r.text, vocab, max_len)
            if not ids:
                continue
            self.embrows.append(lut[key])
            self.ids.append(ids)
        self.emb = emb
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        v = torch.from_numpy(np.asarray(self.emb[self.embrows[i]], dtype=np.float32))
        toks = self.ids[i]
        t = torch.zeros(self.max_len, dtype=torch.long)
        t[:len(toks)] = torch.tensor(toks)
        return v, t, len(toks)


def collate(batch):
    v = torch.stack([b[0] for b in batch])
    t = torch.stack([b[1] for b in batch])
    n = torch.tensor([b[2] for b in batch])
    return v, t, n


# ---------------- model ----------------
class TwoTower(nn.Module):
    def __init__(self, vocab_size, dim=512, drop=0.1):
        super().__init__()
        self.vproj = nn.Sequential(nn.LayerNorm(EMB_DIM), nn.Dropout(drop),
                                   nn.Linear(EMB_DIM, dim))
        self.word = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode_image(self, v):
        return F.normalize(self.vproj(v), dim=-1)

    def encode_text(self, t, n):
        e = self.word(t)                      # [B, L, D]
        mask = (t != 0).unsqueeze(-1).float()
        pooled = (e * mask).sum(1) / n.clamp(min=1).unsqueeze(-1).float()
        return F.normalize(pooled, dim=-1)

    def forward(self, v, t, n):
        iv, tv = self.encode_image(v), self.encode_text(t, n)
        scale = self.logit_scale.clamp(max=np.log(100)).exp()
        logits = scale * iv @ tv.t()
        labels = torch.arange(len(v), device=v.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# ---------------- 4AFC eval ----------------
def eval_4afc(model, emb, lut, eval_frames, vocab, device, n_trials=100, seed=0, max_len=16):
    rng = np.random.default_rng(seed)
    # category -> list of emb-cache rows; and category token ids
    has_key = "key" in eval_frames.columns
    pools, cat_ids = {}, {}
    for cat, g in eval_frames.groupby("category"):
        ids = encode(cat, vocab, max_len)
        keys = g.key if has_key else [frame_key(v, f) for v, f in zip(g.video_id, g.frame_idx)]
        rows = [lut[k] for k in keys if k in lut]
        if ids and len(rows) >= 4:
            pools[cat] = rows
            cat_ids[cat] = ids
    cats = sorted(pools)
    if len(cats) < 4:
        return {"acc": float("nan"), "n_cats": len(cats)}

    model.eval()
    # precompute image projections for all eval rows
    all_rows = sorted({r for rs in pools.values() for r in rs})
    with torch.no_grad():
        V = torch.from_numpy(np.asarray(emb[all_rows], dtype=np.float32)).to(device)
        Vp = model.encode_image(V)
    row2i = {r: i for i, r in enumerate(all_rows)}

    per_cat = {}
    for cat in cats:
        toks = cat_ids[cat]
        t = torch.zeros(1, max_len, dtype=torch.long, device=device)
        t[0, :len(toks)] = torch.tensor(toks, device=device)
        with torch.no_grad():
            tv = model.encode_text(t, torch.tensor([len(toks)], device=device))  # [1,D]
        others = [c for c in cats if c != cat]
        correct = 0
        for _ in range(n_trials):
            tgt = rng.choice(pools[cat])
            dcats = rng.choice(others, size=3, replace=False)
            dist = [rng.choice(pools[c]) for c in dcats]
            cand = [tgt] + dist
            cv = Vp[[row2i[r] for r in cand]]           # [4, D]
            sims = (tv @ cv.t()).squeeze(0)             # [4]
            if sims.argmax().item() == 0:
                correct += 1
        per_cat[cat] = correct / n_trials
    return {"acc": float(np.mean(list(per_cat.values()))),
            "n_cats": len(cats), "per_cat": per_cat}


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--eval-frames", default=str(MANIFEST_DIR / "eval_frames.parquet"))
    ap.add_argument("--emb-dir", default=str(EMB_DIR), help="embedding cache for training pairs")
    ap.add_argument("--eval-emb-dir", default=None, help="embedding cache for eval (default: --emb-dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    emb, lut = load_emb_cache(args.emb_dir)
    eemb, elut = (emb, lut) if not args.eval_emb_dir else load_emb_cache(args.eval_emb_dir)
    man = pd.read_parquet(args.manifest)
    eval_frames = pd.read_parquet(args.eval_frames)

    vocab = build_vocab(man.text, args.min_freq)
    save_json(vocab, outdir / "vocab.json")

    ds = Pairs(emb, lut, man, vocab)
    print(f"usable pairs (frame embedded + tokens in vocab): {len(ds)} | vocab {len(vocab)}",
          flush=True)
    n_val = max(1, int(len(ds) * args.val_frac))
    tr, va = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))
    dl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                     drop_last=True, collate_fn=collate, num_workers=0)
    vdl = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False,
                                      collate_fn=collate, num_workers=0)

    model = TwoTower(len(vocab), args.dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    log = []
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for v, t, n in dl:
            v, t, n = v.to(dev), t.to(dev), n.to(dev)
            loss = model(v, t, n)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        tl /= max(1, len(dl))

        model.eval(); vl = 0.0
        with torch.no_grad():
            for v, t, n in vdl:
                v, t, n = v.to(dev), t.to(dev), n.to(dev)
                vl += model(v, t, n).item()
        vl /= max(1, len(vdl))

        ev = eval_4afc(model, eemb, elut, eval_frames, vocab, dev, seed=args.seed)
        rec = {"epoch": ep, "train_loss": round(tl, 4), "val_loss": round(vl, 4),
               "acc_4afc": round(ev["acc"], 4), "n_eval_cats": ev["n_cats"]}
        log.append(rec)
        print(rec, flush=True)

    save_json(log, outdir / "log.json")
    save_json(ev.get("per_cat", {}), outdir / "per_cat_final.json")
    torch.save(model.state_dict(), outdir / "model.pt")
    print("DONE", outdir, flush=True)


if __name__ == "__main__":
    main()
