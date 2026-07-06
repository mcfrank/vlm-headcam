"""Confusion structure of the best organic model: for each category word, which categories' images
does it score highest? Reveals whether failures are coherent visual/semantic confusions. Test-60."""
import json
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_region_mil import RegionMIL
from train import encode
from common import frame_key

W = "/data2/mcfrank/vlm-headcam"
run = sys.argv[1] if len(sys.argv) > 1 else "G_base_mil_full_s0"
dev = "cuda" if torch.cuda.is_available() else "cpu"
vocab = json.load(open(f"{W}/runs/{run}/vocab.json"))
m = RegionMIL(len(vocab)).to(dev)
m.load_state_dict(torch.load(f"{W}/runs/{run}/model.pt", map_location=dev)); m.eval()

idx = pd.read_parquet(f"{W}/emb_konkle/index.parquet")
emb = np.load(f"{W}/emb_konkle/emb.f16.npy", mmap_mode="r")
lut = {frame_key(v, f): int(r) for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row)}
ev = pd.read_parquet(f"{W}/manifests/eval_frames_konkle.parquet")

cats, img_rows, img_cat = [], [], []
for cat, g in ev.groupby("category"):
    if not encode(cat, vocab, 16):
        continue
    rows = [lut[frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx) if frame_key(v, f) in lut]
    if len(rows) >= 4:
        cats.append(cat); img_rows += rows; img_cat += [cat] * len(rows)
cats = sorted(set(cats))
V = torch.from_numpy(np.asarray(emb[img_rows], dtype=np.float32)).to(dev)
Rp = m.enc_regions(V)                                                        # [K,R,D]
TV = torch.stack([m.enc_text(torch.tensor([encode(c, vocab, 16) + [0] * (16 - len(encode(c, vocab, 16)))], device=dev),
                             torch.tensor([len(encode(c, vocab, 16))], device=dev))[0] for c in cats])  # [C,D]
with torch.no_grad():
    sims = torch.einsum('krd,cd->krc', Rp, TV).max(1).values.cpu().numpy()    # [K,C] image-score per word
ic = np.array(img_cat)
# S[word i, cat j] = mean score of cat-j images under word-i
S = np.array([[sims[ic == cj, i].mean() for cj in cats] for i, _ in enumerate(cats)])

# per word: rank of the correct category, and top confusion
conf = []
for i, c in enumerate(cats):
    order = np.argsort(-S[i])
    rank = int(np.where(order == i)[0][0])
    j = order[0] if order[0] != i else order[1]
    conf.append((c, rank, cats[j], S[i, j] - S[i, i]))
cf = pd.DataFrame(conf, columns=["category", "self_rank", "top_confused_with", "margin"]).sort_values("margin", ascending=False)
print("=== strongest confusions (word prefers another category's images) ===")
print(cf.head(14).to_string(index=False))

# heatmap of the 24 most-confused categories (row-normalized)
worst = cf.head(24).category.tolist(); wi = [cats.index(c) for c in worst]
Sw = S[np.ix_(wi, wi)]; Sw = (Sw - Sw.min(1, keepdims=True)) / (np.ptp(Sw, axis=1, keepdims=True) + 1e-9)
fig, ax = plt.subplots(figsize=(9, 8), dpi=140)
im = ax.imshow(Sw, cmap="magma")
ax.set_xticks(range(len(worst))); ax.set_xticklabels(worst, rotation=90, fontsize=7)
ax.set_yticks(range(len(worst))); ax.set_yticklabels(worst, fontsize=7)
ax.set_xlabel("images of category", fontsize=10); ax.set_ylabel("cued by word", fontsize=10)
ax.set_title("Confusion (row-normalized score) — 24 most-confused categories", fontsize=11, loc="left")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(f"{W}/book_figs/fig_confusion.png", bbox_inches="tight")
print("wrote fig_confusion.png")
