"""What got learned — interpretation. (1) What predicts an item's accuracy: training frequency vs
visual distinctiveness. (2) Vision-vs-text bottleneck: a frozen-feature PROTOTYPE 4AFC (no learned
text — pick the image closest to the category's mean DINOv2 embedding) marks what the features can
separate; model < prototype means the failure is in the learned text map, not the features."""
from collections import Counter
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import tokenize, frame_key

import sys
W = "/data2/mcfrank/vlm-headcam"
run = sys.argv[1] if len(sys.argv) > 1 else "G_framereg_s0"
acc = pd.read_parquet(f"{W}/book_figs/item_acc_{run}.parquet")   # category, acc, set

# frequency: # training pairs whose utterance contains the category word
G = pd.read_parquet(f"{W}/scored/gemini_full.parquet")
wc = Counter()
for txt in G.text.astype(str):
    for tk in set(tokenize(txt)):
        wc[tk] += 1
acc["freq"] = acc.category.map(lambda c: wc.get(c, 0))


def proto_acc(cd, ev, n=300):
    emb = np.load(f"{cd}/emb.f16.npy", mmap_mode="r")
    idx = pd.read_parquet(f"{cd}/index.parquet")
    lut = {frame_key(v, f): int(r) for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row)}
    pools = {}
    for cat, g in ev.groupby("category"):
        rows = [lut[frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx) if frame_key(v, f) in lut]
        if len(rows) >= 4:
            X = np.asarray(emb[rows][:, 0, :], dtype=np.float32)          # CLS token
            pools[cat] = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cats = sorted(pools)
    cent = {c: pools[c].mean(0) / (np.linalg.norm(pools[c].mean(0)) + 1e-9) for c in cats}
    rng = np.random.default_rng(0); out = {}
    for cat in cats:
        others = [c for c in cats if c != cat]; correct = 0
        for _ in range(n):
            tgt = pools[cat][rng.integers(len(pools[cat]))]
            dist = [pools[c][rng.integers(len(pools[c]))] for c in rng.choice(others, 3, replace=False)]
            cand = np.stack([tgt] + dist)
            correct += int((cand @ cent[cat]).argmax() == 0)
        out[cat] = correct / n
    return out


proto = {}
for cd, evf in [("emb_konkle", "eval_frames_konkle"), ("emb_konkle_dev", "eval_frames_konkle_dev")]:
    proto.update(proto_acc(f"{W}/{cd}", pd.read_parquet(f"{W}/manifests/{evf}.parquet")))
acc["proto"] = acc.category.map(lambda c: proto.get(c, np.nan)) * 100
a = acc.dropna(subset=["proto"])

print(f"n={len(a)} categories")
print(f"vision-prototype 4AFC: median {a.proto.median():.0f}, min {a.proto.min():.0f}, "
      f"frac>=90%: {(a.proto>=90).mean()*100:.0f}%")
print(f"model-acc vs log-frequency : spearman {spearmanr(a.acc, np.log1p(a.freq)).correlation:.3f}")
print(f"model-acc vs vision-prototype: spearman {spearmanr(a.acc, a.proto).correlation:.3f}")
gap = a[a.proto - a.acc > 25].sort_values("proto", ascending=False)
print(f"learnable-by-vision-but-model-missed (proto-model>25): {len(gap)} cats, e.g.",
      gap.head(10).category.tolist())
both_low = a[(a.proto < 45) & (a.acc < 40)]
print(f"vision-feature-limited (both low): {len(both_low)} cats, e.g.", both_low.head(10).category.tolist())
a.to_parquet(f"{W}/book_figs/item_analysis.parquet", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), dpi=140)
ax[0].hist(a.proto, bins=np.linspace(20, 100, 33), color="#1d9e75")
ax[0].axvline(a.acc.mean(), color="#b0655a", lw=1.5, ls="--")
ax[0].text(a.acc.mean() - 1, ax[0].get_ylim()[1] * 0.8, f"model mean {a.acc.mean():.0f}", color="#b0655a", ha="right", fontsize=9)
ax[0].set_xlabel("vision prototype 4AFC (frozen features)"); ax[0].set_ylabel("# categories")
ax[0].set_title("Frozen features separate every category", fontsize=11, loc="left")
ax[1].scatter(np.log1p(a.freq), a.acc, s=22, color="#185fa5")
ax[1].axhline(25, color="#6b6a66", lw=1, ls=(0, (5, 4)))
ax[1].set_xlabel("log(1 + training frequency)"); ax[1].set_ylabel("model 4AFC")
ax[1].set_title(f"Learning tracks frequency (ρ={spearmanr(a.acc, np.log1p(a.freq)).correlation:.2f}), not vision (ρ={spearmanr(a.acc, a.proto).correlation:.2f})", fontsize=10.5, loc="left")
for x in ax:
    for s in ("top", "right"): x.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{W}/book_figs/fig_item_analysis.png", bbox_inches="tight")
print("wrote fig_item_analysis.png")
