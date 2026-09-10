"""Confusion RDM of the best organic model. Restrict to the LEARNED categories (>=50%), build a
symmetric confusion matrix (how much word_C prefers category-D images and vice versa), hierarchically
cluster it, and reorder rows/cols by the clustering to expose confusable clusters."""
import json
import sys
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
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
m.load_state_dict(torch.load(f"{W}/runs/{run}/model.pt", map_location=dev), strict=False); m.eval()  # older ckpts lack center_mu (zeros = off)
learned = set(pd.read_parquet(f"{W}/book_figs/item_acc_{run}.parquet").query("acc >= 50").category)

# gather image region-embeddings for learned categories, across both eval caches
img_rows_by_cat, Rp_all, base = {}, [], 0
Rps = []
for cd, evf in [("emb_konkle", "eval_frames_konkle"), ("emb_konkle_dev", "eval_frames_konkle_dev")]:
    idx = pd.read_parquet(f"{W}/{cd}/index.parquet")
    emb = np.load(f"{W}/{cd}/emb.f16.npy", mmap_mode="r")
    lut = {frame_key(v, f): int(r) for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row)}
    ev = pd.read_parquet(f"{W}/manifests/{evf}.parquet")
    for cat, g in ev.groupby("category"):
        if cat not in learned or not encode(cat, vocab, 16):
            continue
        rows = [lut[frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx) if frame_key(v, f) in lut]
        if len(rows) < 4:
            continue
        Rp = m.enc_regions(torch.from_numpy(np.asarray(emb[rows], dtype=np.float32)).to(dev))
        img_rows_by_cat.setdefault(cat, []).append(Rp)
cats = sorted(img_rows_by_cat)
cat_Rp = {c: torch.cat(img_rows_by_cat[c]) for c in cats}          # [n_img, R, D] per cat

def word_vec(c):
    ids = encode(c, vocab, 16); t = torch.zeros(1, 16, dtype=torch.long, device=dev); t[0, :len(ids)] = torch.tensor(ids, device=dev)
    return m.enc_text(t, torch.tensor([len(ids)], device=dev))[0]
TV = torch.stack([word_vec(c) for c in cats])                      # [C, D]

# S[i,j] = mean over cat-j images of max-region score for word i
with torch.no_grad():
    S = np.array([[torch.einsum('nrd,d->nr', cat_Rp[cj], TV[i]).max(1).values.mean().item()
                   for cj in cats] for i in range(len(cats))])
M = (S + S.T) / 2                                                  # symmetric confusion
Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)   # row-z for clustering
Dm = 1 - np.corrcoef(Z); np.fill_diagonal(Dm, 0); Dm = (Dm + Dm.T) / 2
L = linkage(squareform(Dm, checks=False), method="average")
order = leaves_list(L)
clusters = fcluster(L, t=9, criterion="maxclust")
print(f"{len(cats)} learned categories | clusters:")
for k in sorted(set(clusters)):
    members = [cats[i] for i in range(len(cats)) if clusters[i] == k]
    if len(members) >= 3:
        print(f"  cluster {k} ({len(members)}): {members}")

np.savez(f"{W}/book_figs/confusion_M.npz", M=M, cats=np.array(cats), order=order, clusters=clusters)
Mo = M[np.ix_(order, order)]; lab = [cats[i] for i in order]
fig, ax = plt.subplots(figsize=(11, 10), dpi=150)
im = ax.imshow(Mo, cmap="magma")
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=90, fontsize=5.5)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=5.5)
ax.set_title("Confusion RDM among learned categories (clustered order)", fontsize=12, loc="left")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(f"{W}/book_figs/fig_confusion.png", bbox_inches="tight")
print("wrote fig_confusion.png")
