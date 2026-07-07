"""LEVANTE-vocab secondary-eval figure: model competence vs children's item difficulty (weak)
vs the model's own BabyView frequency (the real driver). Seed-mean over 3 runs; generous labels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/data2/mcfrank/vlm-headcam"
p = pd.read_parquet(f"{W}/book_figs/lev_vocab_seedmean_freq.parquet")
INK, SUB, GRID, Gc, Rc = "#2c2c2a", "#6b6a66", "#e1e0d9", "#1d9e75", "#b0655a"
p["col"] = np.where(p.correct >= 0.5, Gc, Rc)
rho_d = p[["p_correct", "d"]].corr("spearman").iloc[0, 1]
rho_f = p[["p_correct", "logfreq"]].corr("spearman").iloc[0, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=150)


def panel(ax, x, y, title, xlabel, fit=True):
    ax.scatter(x, y, c=p.col, s=30, alpha=.85, zorder=3, edgecolor="white", linewidth=.5)
    if fit:
        b, a = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color=SUB, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.axhline(.25, color=SUB, lw=1, ls=(0, (5, 4)))
    for _, r in p.iterrows():
        ax.annotate(r.target_word, (x[r.name], y[r.name]), fontsize=6.3, color=INK,
                    xytext=(2.5, 2), textcoords="offset points", zorder=4)
    ax.set_xlabel(xlabel, fontsize=10, color=SUB)
    ax.set_ylabel("model P(correct image)", fontsize=10, color=SUB)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SUB, length=0); ax.yaxis.grid(True, color=GRID, lw=.7); ax.set_axisbelow(True)


panel(ax1, p.d, p.p_correct, f"A. vs children's item difficulty   (ρ = {rho_d:+.2f}, weak)",
      "child item difficulty  (IRT d; harder →)")
ax1.text(p.d.min(), .265, "chance", color=SUB, fontsize=8, va="bottom")
panel(ax2, p.logfreq, p.p_correct, f"B. vs the model's own BabyView frequency   (ρ = {rho_f:+.2f})",
      "log$_{10}$ BabyView training frequency")
ax2.text(p.logfreq.min(), .265, "chance", color=SUB, fontsize=8, va="bottom")
fig.suptitle("LEVANTE vocabulary: the model's competence tracks its own input frequency, "
             "not children's difficulty", fontsize=12.5, color=INK, x=.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, .96])
fig.savefig(f"{W}/book_figs/fig_lev_vocab.png", bbox_inches="tight")
print(f"wrote fig_lev_vocab.png | n={len(p)} rho_d={rho_d:.3f} rho_f={rho_f:.3f}")
