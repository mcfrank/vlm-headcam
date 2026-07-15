"""Frame-MIL: an utterance matches its best (frame, region) over a ±W-second window, not just
the midpoint frame. Extends region-MIL's max-over-cells to a max over frames × cells by stacking
the window frames' region grids and letting the same RegionMIL.sims max over all of them.

Cells of the region×frame 2×2:
  region-only  = region-MIL baseline (window 0)                    -> already have (G_base_mil)
  whole-frame-only = --cls-only --window 0                          -> ~ the no-MIL baseline
  frame-only   = --cls-only --window 2  (max over frames' CLS)
  region+frame = --window 2             (max over frames × regions)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from common import frame_key, save_json
from train import build_vocab, encode
from train_region_mil import RegionMIL, eval_4afc_region, collate, load_region_cache


def load_multi(dirs):
    """Union several region caches into one key -> (cache_idx, row) lut; caches stay mmap'd."""
    caches, lut = [], {}
    for ci, d in enumerate(dirs):
        idx = pd.read_parquet(Path(d) / "index.parquet")
        caches.append(np.load(Path(d) / "emb.f16.npy", mmap_mode="r"))
        for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row):
            lut[frame_key(v, int(f))] = (ci, int(r))
    return caches, lut


class FramePairs(torch.utils.data.Dataset):
    def __init__(self, caches, lut, man, vocab, window, cls_only, max_len=16):
        self.caches, self.window, self.cls_only, self.max_len = caches, window, cls_only, max_len
        self.Fmax = 2 * window + 1
        self.locs, self.ids = [], []
        for r in man.itertuples(index=False):
            toks = encode(r.text, vocab, max_len)
            if not toks:
                continue
            fi = int(r.frame_idx)
            here = [lut[frame_key(r.video_id, f)] for f in range(fi - window, fi + window + 1)
                    if frame_key(r.video_id, f) in lut]
            if not here:
                continue
            self.locs.append(here); self.ids.append(toks)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        grids = [np.asarray(self.caches[ci][row], dtype=np.float32) for ci, row in self.locs[i]]
        while len(grids) < self.Fmax:                 # pad by repeat (max-pool ignores dups)
            grids.append(grids[0])
        v = np.stack(grids)                            # [Fmax, R, 768]
        if self.cls_only:
            v = v[:, 0:1, :]
        v = v.reshape(-1, v.shape[-1])                 # [Fmax*R', 768]
        t = torch.zeros(self.max_len, dtype=torch.long)
        t[:len(self.ids[i])] = torch.tensor(self.ids[i])
        return i, torch.from_numpy(v), t, len(self.ids[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--eval-frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--cls-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=None, help="encoder feature dim (default: cache dim)")
    ap.add_argument("--center", action="store_true", help="subtract train dataset-mean before projection (anisotropy fix)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    man = pd.read_parquet(a.manifest)
    vocab = build_vocab(man.text, 5)
    caches, lut = load_multi(a.caches)
    emb_dim = a.emb_dim or caches[0].shape[-1]
    ds = FramePairs(caches, lut, man, vocab, a.window, a.cls_only)
    print(f"pairs {len(ds)} | vocab {len(vocab)} | window +-{a.window} | cls_only {a.cls_only} | emb_dim {emb_dim}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     collate_fn=collate, num_workers=4)
    m = RegionMIL(len(vocab), emb_dim=emb_dim).to(dev)
    if a.center:
        mu = np.asarray(caches[0]).reshape(-1, emb_dim).mean(0)
        m.center_mu.copy_(torch.from_numpy(mu.astype(np.float32)).to(dev))
        print(f"centered: subtracted train mean |mu|={np.linalg.norm(mu):.2f}", flush=True)
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
