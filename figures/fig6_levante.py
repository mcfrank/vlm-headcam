"""Display item 6 (pocket) — the external check against real children.

LEVANTE-bench vocabulary is a 4AFC over the same shape as our Konkle eval, but with IRT item
difficulties estimated from ~1,500 children aged 5-12. It is the only place actual children appear.

A: where the model lands among generative VLMs, with children as a reference band.
B: the item-level story — the model's competence tracks the word's frequency in ITS OWN input
   far more than it tracks how hard the item is for children.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
it = pd.read_parquet(R / "lev_vocab_seedmean.parquet")
fq = pd.read_parquet(R / "lev_vocab_seedmean_freq.parquet")

# scored fairly across all 159: real accuracy on the 100 attemptable, chance on the 59 OOV
known = it.correct.notna()
ours_fair = 100 * (it.loc[known, "correct"].sum() + 0.25 * (~known).sum()) / len(it)
MODELS = [("ours\n~0.5M", ours_fair, T.FREE), ("TinyLLaVA\n3.1B", 42, T.NEUTRAL),
          ("InternVL3.5\n1B", 50, T.NEUTRAL), ("Qwen3.5\n0.8B", 72, T.OTHER),
          ("Gemma4\n4B", 93, T.OTHER), ("GPT-5.3\nfrontier", 99, T.ORACLE)]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.5),
                             gridspec_kw=dict(width_ratios=[1, 1.1]))

# ---- A ---------------------------------------------------------------------------
xs = np.arange(len(MODELS))
ax.axhspan(72, 82, color=T.CHILD, alpha=0.30, lw=0, zorder=1)
ax.text(len(MODELS) - 0.4, 77, "children 5–12 yr", fontsize=5.8, color="#8a6d1f",
        ha="right", va="center")
ax.bar(xs, [m[1] for m in MODELS], color=[m[2] for m in MODELS], width=0.62, zorder=3)
for x_, (lab, v, _) in zip(xs, MODELS):
    ax.text(x_, v + 1.5, f"{v:.0f}", ha="center", fontsize=6, color=T.INK)
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(len(MODELS) - 0.4, 27, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.set_xticks(xs); ax.set_xticklabels([m[0] for m in MODELS], fontsize=5.6)
ax.set_ylabel("LEVANTE vocabulary 4AFC (%)"); ax.set_ylim(0, 108)
ax.get_xticklabels()[0].set_color(T.FREE)
T.clean(ax)

# ---- B: difficulty vs frequency ---------------------------------------------------
d = fq.dropna(subset=["p_correct", "d", "logfreq"])
rho_child = spearmanr(d.p_correct, d.d).correlation
rho_freq = spearmanr(d.p_correct, d.logfreq).correlation
sc = bx.scatter(d.logfreq, 100 * d.p_correct, s=13, c=d.d, cmap="cividis",
                edgecolors="none", zorder=3)
m, b = np.polyfit(d.logfreq, 100 * d.p_correct, 1)
xx = np.linspace(d.logfreq.min(), d.logfreq.max(), 20)
bx.plot(xx, m * xx + b, color=T.INK, lw=1.0, zorder=4)
bx.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.set_xlabel("log frequency in the model's own BabyView input")
bx.set_ylabel("model confidence on the item (%)")
bx.text(0.03, 0.95, f"vs own-input frequency   ρ = {rho_freq:+.2f}\n"
        f"vs children's difficulty   ρ = {rho_child:+.2f}",
        transform=bx.transAxes, fontsize=6, va="top", color=T.INK)
cb = fig.colorbar(sc, ax=bx, fraction=0.04, pad=0.02)
cb.set_label("children's IRT difficulty", fontsize=5.8, color=T.SUB)
cb.ax.tick_params(labelsize=5.4)
T.clean(bx)
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
print(f"  NOTE fig6: A scores ours across all {len(it)} items ({known.sum()} attemptable, chance on "
      f"{(~known).sum()}); other models prompted generatively. B: {len(d)} items with a child difficulty.")
T.save(fig, "fig6_levante")
