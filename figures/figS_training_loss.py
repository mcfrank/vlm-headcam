"""DINOv3 pre-training loss curves for the three BabyView-trained encoders.

Total self-distillation loss (DINO global + local + KoLeo + iBOT) over the full 200k-
iteration schedule, one curve per model size. All three runs used the identical schedule
and the FINAL checkpoint (iteration 200k) everywhere in the paper — probes, embeddings,
released weights — marked here; no checkpoint selection was performed.

Data: results/dino_training_curves.parquet, distilled from each run's training_metrics.json
(logged every 10 iterations; slice-resume overlaps deduplicated keeping the last pass).
Loss values are comparable across runs in shape, not level: absolute DINO loss depends on
head/prototype dimensionality. BV-304M ran at global batch 384 (the others 512) with the
LR rule's sqrt scaling, so equal iterations are ~25% fewer samples for that run.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
d = pd.read_parquet(R / "dino_training_curves.parquet")
CKPT = 199_999
ENC = [T.enc(t) for t in ("vitl_bv", "vitb_bv", "vits_bv")]

fig, ax = plt.subplots(figsize=(T.W1, 2.3))
ends = []
for E in ENC:
    g = d[d.model == E["tag"]].sort_values("iteration")
    sm = g.total_loss.rolling(51, center=True, min_periods=1).mean()
    ax.plot(g.iteration, g.total_loss, color=E["color"], lw=0.4, alpha=0.22, zorder=1)
    ax.plot(g.iteration, sm, color=E["color"], lw=1.0, zorder=2)
    ax.scatter([CKPT], [sm.iloc[-1]], s=11, color=E["color"], zorder=4,
               edgecolors="white", lw=0.5)
    ends.append((float(sm.iloc[-1]), E["label"], E["color"]))

ax.axvline(CKPT, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=0)
ax.text(CKPT - 4e3, 17.9, "selected checkpoint\n(200k, final — all models)",
        fontsize=5.2, color=T.SUB, ha="right", va="top")
T.end_labels(ax, [CKPT] * len(ends), [e[0] for e in ends], [e[1] for e in ends],
             [e[2] for e in ends], gap=0.55, xl=CKPT + 6e3)

ax.set_xlim(0, 245_000)
ax.set_xticks([0, 50_000, 100_000, 150_000, 200_000])
ax.set_xticklabels(["0", "50k", "100k", "150k", "200k"])
ax.set_xlabel("training iteration")
ax.set_ylabel("total DINOv3 loss")
T.clean(ax)
T.save(fig, "figS_training_loss")
