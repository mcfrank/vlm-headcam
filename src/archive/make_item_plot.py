"""Per-category (item) accuracy for the best organic model, Vong-2024 style: which words got
learned. Also dumps per_cat to a parquet for downstream interpretation. Aggregate-only."""
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
from common import frame_key, load_emb_cache

W = "/data2/mcfrank/vlm-headcam"
run = sys.argv[1] if len(sys.argv) > 1 else "G_framereg_s0"
dev = "cuda" if torch.cuda.is_available() else "cpu"
vocab = json.load(open(f"{W}/runs/{run}/vocab.json"))
m = RegionMIL(len(vocab)).to(dev)
m.load_state_dict(torch.load(f"{W}/runs/{run}/model.pt", map_location=dev)); m.eval()


def load_region_cache(d):
    idx = pd.read_parquet(f"{d}/index.parquet")
    emb = np.load(f"{d}/emb.f16.npy", mmap_mode="r")
    lut = {frame_key(v, f): int(r) for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row)}
    return emb, lut


def per_cat(emb, lut, ev, n_trials=300, seed=0):
    rng = np.random.default_rng(seed)
    pools, cat_ids = {}, {}
    for cat, g in ev.groupby("category"):
        ids = encode(cat, vocab, 16)
        rows = [lut[frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx) if frame_key(v, f) in lut]
        if ids and len(rows) >= 4:
            pools[cat], cat_ids[cat] = rows, ids
    cats = sorted(pools)
    all_rows = sorted({r for rs in pools.values() for r in rs})
    V = torch.from_numpy(np.asarray(emb[all_rows], dtype=np.float32)).to(dev)
    Rp = m.enc_regions(V); r2i = {r: i for i, r in enumerate(all_rows)}
    out = {}
    for cat in cats:
        toks = cat_ids[cat]; t = torch.zeros(1, 16, dtype=torch.long, device=dev); t[0, :len(toks)] = torch.tensor(toks, device=dev)
        tv = m.enc_text(t, torch.tensor([len(toks)], device=dev))
        others = [c for c in cats if c != cat]; correct = 0
        for _ in range(n_trials):
            cand = [rng.choice(pools[cat])] + [rng.choice(pools[c]) for c in rng.choice(others, 3, replace=False)]
            sc = torch.einsum('brd,md->brm', Rp[[r2i[r] for r in cand]], tv).max(1).values.squeeze(-1)
            correct += int(sc.argmax().item() == 0)
        out[cat] = correct / n_trials
    return out


pc = {}
for d, evf, tag in [(f"{W}/emb_konkle", f"{W}/manifests/eval_frames_konkle.parquet", "test"),
                    (f"{W}/emb_konkle_dev", f"{W}/manifests/eval_frames_konkle_dev.parquet", "dev")]:
    emb, lut = load_region_cache(d)
    for c, a in per_cat(emb, lut, pd.read_parquet(evf)).items():
        pc[c] = (a, tag)

df = pd.DataFrame([(c, a * 100, t) for c, (a, t) in pc.items()], columns=["category", "acc", "set"]).sort_values("acc", ascending=False)
df.to_parquet(f"{W}/book_figs/item_acc_{run}.parquet", index=False)
print(f"{run}: {len(df)} categories | mean {df.acc.mean():.1f} | >=50%: {(df.acc>=50).mean()*100:.0f}% | best/worst")
print("  BEST:", df.head(12).set_index("category").acc.round(0).to_dict())
print("  WORST:", df.tail(12).set_index("category").acc.round(0).to_dict())

fig, ax = plt.subplots(figsize=(13, 4), dpi=140)
colors = ["#1d9e75" if a >= 50 else ("#e0b03a" if a >= 37.5 else "#b0655a") for a in df.acc]
ax.bar(range(len(df)), df.acc, color=colors, width=0.9)
ax.axhline(25, color="#6b6a66", lw=1, ls=(0, (5, 4))); ax.text(len(df) - 1, 26, "chance", color="#6b6a66", fontsize=9, ha="right")
ax.set_xlim(-1, len(df)); ax.set_ylim(0, 100); ax.set_ylabel("4AFC accuracy", fontsize=11)
ax.set_title(f"What got learned — per-category accuracy ({len(df)} Konkle categories)", fontsize=12, loc="left")
step = max(1, len(df) // 45)
ax.set_xticks(range(0, len(df), step)); ax.set_xticklabels(df.category[::step], rotation=90, fontsize=6.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{W}/book_figs/fig_item_acc.png", bbox_inches="tight")
print("wrote fig_item_acc.png")
