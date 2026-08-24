"""JOINT encoder + tower training: the CVCL-style counterpart to the frozen rig.

Same head (RegionMIL), same loss (forward_loss), same eval (eval_4afc_region on Konkle
test-60 with dev-117 epoch selection), same 4x4 drop-CLS region grid — the ONLY change is
that gradients flow into the top --unfreeze-blocks of DINOv3. Eval images are re-embedded
through the current encoder every epoch. An encoder-quality probe (leave-one-out prototype
4AFC on pooled features, no head involved) runs at init and at the end, so fine-tuning
damage to the representation is measured, not guessed.

usage: python src/train_joint.py --manifest manifests/bv26_rand_300000_s0.parquet \
         --frames224 /data2/mcfrank/frames224 --seed 0 --out runs/B26J_300000_s0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from common import frame_key, save_json
from train_frame_mil import build_vocab, encode
from train_region_mil import RegionMIL, eval_4afc_region

G = 4


def grid_features(model, px, nreg):
    """pixel batch -> [B, 16, 768] drop-CLS 4x4 grid — byte-for-byte the embed_regions readout."""
    h = model(pixel_values=px).last_hidden_state
    patch = h[:, 1 + nreg:]
    s = int(round(patch.shape[1] ** 0.5))
    assert s * s == patch.shape[1], f"patch count {patch.shape[1]} not square"
    grid = patch.transpose(1, 2).reshape(len(px), -1, s, s)
    pooled = F.adaptive_avg_pool2d(grid, (G, G))
    return pooled.flatten(2).transpose(1, 2)


class JointPairs(torch.utils.data.Dataset):
    def __init__(self, man, root, vocab, mean, std, max_len=16):
        self.root, self.mean, self.std = Path(root), mean, std
        self.rows, self.ids = [], []
        miss = novocab = 0
        for r in man.itertuples(index=False):
            toks = encode(r.text, vocab, max_len)
            if not toks:
                novocab += 1; continue
            p = self.root / r.video_id / f"{int(r.frame_idx):06d}.jpg"
            if not p.exists():
                miss += 1; continue
            self.rows.append(p); self.ids.append(toks)
        self.cov = len(self.rows) / max(len(man), 1)
        print(f"pairs {len(self.rows):,} | manifest {len(man):,} | coverage {100*self.cov:.1f}% "
              f"({novocab:,} no-vocab, {miss:,} frame not in 224 cache)", flush=True)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        im = torch.from_numpy(np.asarray(Image.open(self.rows[i]), dtype=np.float32) / 255.0)
        im = (im.permute(2, 0, 1) - self.mean) / self.std
        return im, self.ids[i]


def collate(b):
    ims = torch.stack([x[0] for x in b])
    L = max(len(x[1]) for x in b)
    t = torch.zeros(len(b), L, dtype=torch.long)
    n = torch.tensor([len(x[1]) for x in b])
    for i, (_, toks) in enumerate(b):
        t[i, :len(toks)] = torch.tensor(toks)
    return ims, t, n


def embed_eval(model, paths, nreg, mean, std, dev, bs=64):
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i in range(0, len(paths), bs):
            ims = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((224, 224)),
                                               dtype=np.float32) / 255.0).permute(2, 0, 1) for p in paths[i:i + bs]]
            px = ((torch.stack(ims) - mean) / std).to(dev)
            out.append(grid_features(model, px, nreg).float().cpu())
    return torch.cat(out).numpy()


def proto_probe(emb, cats, seed=0, n_trials=100):
    """Encoder quality without the head: pooled feature, leave-one-out prototype 4AFC."""
    x = torch.from_numpy(emb.mean(1)); x = F.normalize(x, dim=-1)
    rng = np.random.default_rng(seed)
    bycat = {}
    for i, c in enumerate(cats): bycat.setdefault(c, []).append(i)
    ok = tot = 0
    for c, idx in bycat.items():
        others = [k for k in bycat if k != c]
        for i in idx[:n_trials]:
            proto_c = F.normalize(x[[j for j in idx if j != i]].mean(0), dim=-1)
            foils = rng.choice(others, 3, replace=False)
            pf = torch.stack([proto_c] + [F.normalize(x[bycat[f]].mean(0), dim=-1) for f in foils])
            ok += int((x[i] @ pf.T).argmax().item() == 0); tot += 1
    return 100 * ok / tot


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--frames224", default="/data2/mcfrank/frames224")
ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
ap.add_argument("--eval-manifest", default="manifests/konkle_manifest.parquet")
ap.add_argument("--dev-manifest", default="manifests/eval_frames_konkle_dev.parquet")
ap.add_argument("--unfreeze-blocks", type=int, default=4)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=192)
ap.add_argument("--head-lr", type=float, default=1e-3)
ap.add_argument("--enc-lr", type=float, default=1e-5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min-coverage", type=float, default=0.9)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"

proc = AutoImageProcessor.from_pretrained(a.model)
MEAN = torch.tensor(proc.image_mean).view(3, 1, 1)
STD = torch.tensor(proc.image_std).view(3, 1, 1)
enc = AutoModel.from_pretrained(a.model).to(dev)
NREG = getattr(enc.config, "num_register_tokens", 0) or 0
NL = enc.config.num_hidden_layers
for p in enc.parameters(): p.requires_grad = False
unfrozen = [n for n, p in enc.named_parameters()
            if any(f"layer.{i}." in n for i in range(NL - a.unfreeze_blocks, NL)) or n.startswith("layernorm")]
for n, p in enc.named_parameters():
    if n in unfrozen: p.requires_grad = True
print(f"encoder {a.model}: {NL} layers, unfreezing last {a.unfreeze_blocks} "
      f"({sum(p.numel() for p in enc.parameters() if p.requires_grad)/1e6:.1f}M of "
      f"{sum(p.numel() for p in enc.parameters())/1e6:.1f}M params) | registers {NREG}", flush=True)

man = pd.read_parquet(a.manifest)
vocab = build_vocab(man.text, 5)
ds = JointPairs(man, a.frames224, vocab, MEAN, STD)
if ds.cov < a.min_coverage:
    raise SystemExit(f"ABORT: 224-cache coverage {100*ds.cov:.1f}% < {100*a.min_coverage:.0f}%")
dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                 collate_fn=collate, num_workers=8, pin_memory=True)

head = RegionMIL(len(vocab), emb_dim=enc.config.hidden_size).to(dev)
opt = torch.optim.AdamW([
    {"params": [p for p in enc.parameters() if p.requires_grad], "lr": a.enc_lr},
    {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.1)

ev = pd.read_parquet(a.eval_manifest); dv = pd.read_parquet(a.dev_manifest)
def eval_now():
    enc.eval()
    E = embed_eval(enc, list(ev.path), NREG, MEAN, STD, dev)
    D = embed_eval(enc, list(dv.path), NREG, MEAN, STD, dev)
    elut = {frame_key(v, int(f)): i for i, (v, f) in enumerate(zip(ev.video_id, ev.frame_idx))}
    dlut = {frame_key(v, int(f)): i for i, (v, f) in enumerate(zip(dv.video_id, dv.frame_idx))}
    acc = eval_4afc_region(head, E, elut, ev, vocab, dev)
    dacc = eval_4afc_region(head, D, dlut, dv, vocab, dev)
    return acc, dacc, E

Path(a.out).mkdir(parents=True, exist_ok=True)
_, _, E0 = eval_now()
probe0 = proto_probe(E0, list(ev.category))
print(f"encoder probe at init (prototype 4AFC, no head): {probe0:.1f}", flush=True)

hist, best_sel, best_ep = [], -1.0, -1
best_state = None
for ep in range(a.epochs):
    enc.train(); head.train()
    for ims, t, n in dl:
        ims, t, n = ims.to(dev, non_blocking=True), t.to(dev), n.to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            v = grid_features(enc, ims, NREG)
            loss = head.forward_loss(v.float(), t, n, torch.ones(len(v), device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    acc, dacc, E = eval_now()
    if dacc > best_sel:
        best_sel, best_ep = dacc, ep
        best_state = ({k: p.detach().cpu().clone() for k, p in head.state_dict().items()},
                      {k: p.detach().cpu().clone() for k, p in enc.state_dict().items()})
    rec = {"ep": ep, "acc": round(acc, 4), "dev": round(dacc, 4)}
    hist.append(rec); print(rec, flush=True)

probe1 = proto_probe(E, list(ev.category))
reported = hist[best_ep]["acc"]
if best_state:
    head.load_state_dict(best_state[0]); enc.load_state_dict(best_state[1])
torch.save(head.state_dict(), Path(a.out) / "head.pt")
torch.save({k: v for k, v in enc.state_dict().items()}, Path(a.out) / "encoder.pt")
save_json({"run": Path(a.out).name, "seed": a.seed, "manifest": a.manifest,
           "joint": True, "unfreeze_blocks": a.unfreeze_blocks,
           "enc_lr": a.enc_lr, "head_lr": a.head_lr, "epochs": a.epochs,
           "n_pairs": len(ds), "vocab": len(vocab), "selection": "dev",
           "selected_epoch": best_ep, "reported_test_acc": round(100 * reported, 2),
           "best_test_acc": round(100 * max(h["acc"] for h in hist), 2),
           "final_test_acc": round(100 * hist[-1]["acc"], 2),
           "probe_proto_init": round(probe0, 2), "probe_proto_final": round(probe1, 2),
           "history": hist}, Path(a.out) / "metrics.json")
print(f"DONE {a.out} selected_ep {best_ep} reported {reported:.4f} "
      f"| probe {probe0:.1f} -> {probe1:.1f}", flush=True)
