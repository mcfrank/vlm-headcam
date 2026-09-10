"""Descriptive statistics + figures for the Data chapter (BabyView 2025.2 + annotations).
Aggregate numbers and plots only — no frames, safe to commit."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import tokenize

OUT = "/data2/mcfrank/vlm-headcam/book_figs"
G = pd.read_parquet("scored/gemini_full.parquet")
G = G[G.alignment.notna()].copy()

# ---- corpus ----
per_vid_len = G.groupby("video_id").frame_idx.max()          # ~video duration (s), lower bound
hours = per_vid_len.sum() / 3600
per_kid = G.groupby("child_id").size()
print("=== corpus ===")
print(f"children {G.child_id.nunique()} | videos {G.video_id.nunique()} | "
      f"utterance-frame pairs {len(G):,} | ~{hours:,.0f} h (>= lower bound)")
print(f"utterances/child: median {per_kid.median():.0f}  IQR [{per_kid.quantile(.25):.0f}, {per_kid.quantile(.75):.0f}]  "
      f"min {per_kid.min()} max {per_kid.max()}")

# ---- utterances ----
G["ntok"] = G.text.map(lambda t: len(tokenize(t)))
vocab = pd.Series(np.concatenate(G.text.map(tokenize).values)).value_counts()
print("=== utterances ===")
print(f"tokens/utterance: median {G.ntok.median():.0f} mean {G.ntok.mean():.1f} | "
      f"vocab (>=5) {int((vocab>=5).sum()):,} | total tokens {int(vocab.sum()):,}")

# ---- annotations ----
print("=== annotations ===")
print(f"Gemini: frac0 {np.mean(G.alignment==0):.2f} | >=50 {np.mean(G.alignment>=50)*100:.1f}% | "
      f">=80 {np.mean(G.alignment>=80)*100:.1f}% | has-referent {np.mean(G.referent.fillna('')!='')*100:.1f}%")
print(f"CLIP: mean {G.clip_score_max.mean():.3f} sd {G.clip_score_max.std():.3f}")
samp = G.sample(200000, random_state=0)
print(f"Gemini vs CLIP (n=200k): spearman {spearmanr(samp.alignment, samp.clip_score_max).correlation:.3f}")

# ---- figures ----
INK, ACC = "#2c2c2a", "#185fa5"
def style(ax, t, xl):
    ax.set_title(t, fontsize=11, color=INK, loc="left"); ax.set_xlabel(xl, fontsize=9.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig, ax = plt.subplots(1, 4, figsize=(15, 3.3), dpi=140)
ax[0].hist(np.clip(per_kid.values, 0, 120000), bins=30, color=ACC); style(ax[0], "Utterances per child", "pairs")
ax[1].hist(np.clip(G.ntok, 0, 30), bins=30, color=ACC); style(ax[1], "Utterance length", "tokens")
ax[2].hist(G.alignment, bins=21, color=ACC); style(ax[2], "Gemini alignment", "0–100"); ax[2].set_yscale("log")
ax[3].hexbin(G.clip_score_max, G.alignment, gridsize=40, cmap="Blues", bins="log")
style(ax[3], "Gemini vs CLIP", "CLIP clip_score_max"); ax[3].set_ylabel("Gemini alignment")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_data_descriptives.png", bbox_inches="tight")
print("wrote", f"{OUT}/fig_data_descriptives.png")
