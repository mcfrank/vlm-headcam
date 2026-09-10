"""Exploratory: the oracle arm and ladder in lexical space (not a paper display item).

A: relatedness vs training pairs for the unfiltered arm and the aligned (oracle) arm,
   each against a word2vec trained on ITS OWN utterances.
B: the oracle ladder at ~170k pairs across all four encoders, with the same-utterance
   word2vec topline. -> scratch figure path given as argv[1].
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "figures"))
import pandas as pd, numpy as np, theme as T
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.7),
                             gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.33))

fr = pd.read_csv(R / "lexicon_ws_scaling_F-dinov3l.csv")
al = pd.read_csv(R / "lexicon_ws_scaling_F-dinov3l-aligned.csv")
for d, col, lab, mk in [(fr, T.FREE, "unfiltered", "o"), (al, T.ORACLE, "aligned (oracle)", "s")]:
    n = d[d.category == "noun"]
    m = n[n.kind == "model"].groupby("scale").agg(y=("spearman", "mean"), e=("spearman", "std"))
    m = m[m.index >= 1e4]
    ax.errorbar(m.index, m.y, yerr=m.e, fmt="-" + mk, color=col, ms=3.2, lw=1.1,
                elinewidth=0.7, capsize=1.8, zorder=3, label=lab)
    w = n[n.kind == "w2v"].groupby("scale").spearman.mean(); w = w[w.index >= 1e4]
    ax.plot(w.index, w.values, "--", color=col, lw=0.8, alpha=0.55, zorder=2)
ax.axhline(0, color=T.GRID, lw=0.6)
ax.set_xscale("log"); ax.set_xlabel("training pairs"); ax.set_ylim(-0.13, 0.46)
ax.set_ylabel("\u03c1 with human relatedness (noun pairs)")
ax.legend(fontsize=6, loc="upper left", frameon=False)
ax.text(0.97, 0.06, "dashed = word2vec on the\nsame utterances", fontsize=5.2, color=T.SUB,
        transform=ax.transAxes, ha="right", linespacing=1.3)
T.clean(ax)

rg = pd.read_csv(R / "lexicon_rungs.csv"); rg = rg[rg.category == "noun"]
ORDER = ["aligned", "filtnat", "word selection", "vision binding"]
ENC = [("dinov3l", T.OTHER, 0.02), ("dinov3b", T.FREE, -0.02),
       ("vitb_bv", T.INDOM, 0.022), ("vits_bv", T.INDOM2, -0.026)]
xs = np.arange(len(ORDER))
for e, col, dy in ENC:
    g = rg[rg.encoder == e].groupby("rung").agg(y=("spearman", "mean"),
                                                e=("spearman", "std")).reindex(ORDER)
    bx.errorbar(xs, g.y, yerr=g.e, fmt="-o", color=col, ms=3.0, lw=1.1, elinewidth=0.7,
                capsize=1.8, zorder=3)
    bx.text(len(ORDER) - 0.85, g.y.iloc[-1] + dy, e.replace("_bv", " BV"), fontsize=5.4,
            color=col, va="center", ha="left")
w2v = rg.groupby("rung").w2v_spearman.mean().reindex(ORDER)
bx.plot(xs, w2v, "--s", color=T.SUB, ms=2.6, lw=0.9, zorder=2)
bx.text(0.05, w2v.iloc[0] + 0.02, "word2vec on the same utterances", fontsize=5.2, color=T.SUB)
bx.axhline(0, color=T.GRID, lw=0.6)
bx.set_xticks(xs)
bx.set_xticklabels(["aligned\nonly", "+ alignment\nfilter", "+ word\nselection",
                    "+ vision\nbinding"], fontsize=5.6)
bx.set_xlim(-0.35, len(ORDER) + 0.25); bx.set_ylim(-0.13, 0.46)
bx.set_ylabel("\u03c1 with human relatedness (noun pairs)")
bx.text(0.03, 0.96, "oracle ladder at ~170k pairs", fontsize=6, transform=bx.transAxes, va="top")
T.clean(bx)
for a_, l in zip((ax, bx), "AB"):
    T.panel(a_, l)
fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight")
print("wrote", sys.argv[1])
