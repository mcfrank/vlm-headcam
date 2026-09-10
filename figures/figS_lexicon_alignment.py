"""what referential selection does to the LEXICON.

Matched-data comparison for every encoder: human-relatedness rho for the aligned arm
(trained only on referential pairs) against the unfiltered arm, on the same x-axis of
training pairs. Alignment buys a far more human-like word space at matched data -- the
same data-efficiency shape fig2 shows on 4AFC, in an independent measure. Noun-noun pairs;
points are seeds, bars are across-seed SD.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
ENC = [(E["tag"], E["lex"], E["label"], E["color"]) for E in T.ENCODERS
       if (R / f"lexicon_ws_scaling_{E['lex']}-aligned.csv").exists()]
missing = [E["label"] for E in T.ENCODERS if E["tag"] not in {e[0] for e in ENC}]
if missing:
    print(f"  NOTE figS_lexicon_alignment: no lexicon extracted yet for {missing}")

fig, ax = plt.subplots(figsize=(T.W15, 2.8))

ends = []
for enc, fam, key, col in ENC:
    for tag, ls, mk, fc in [("", (0, (2.5, 1.5)), "o", "white"), ("-aligned", "-", "o", None)]:
        d = pd.read_csv(R / f"lexicon_ws_scaling_{fam}{tag}.csv")
        d = d[(d.category == "noun") & (d.kind == "model") & (d.scale >= 1e4)]
        g = d.groupby("scale").spearman.agg(["mean", "std"])
        if not tag:
            ends.append((g.index.max(), g["mean"].iloc[-1], key, col))
        ax.errorbar(g.index, g["mean"], yerr=g["std"], color=col, ls=ls, marker=mk, ms=2.8,
                    lw=1.1 if tag else 0.9, elinewidth=0.6, capsize=1.4,
                    markerfacecolor=fc or col, markeredgewidth=0.8,
                    zorder=4 if tag else 3, alpha=1.0 if tag else 0.85)
    if enc == "dinov3l":
        al = pd.read_csv(R / f"lexicon_ws_scaling_{fam}-aligned.csv")
        al = al[(al.category == "noun") & (al.kind == "model")].groupby("scale").spearman.mean()
        un = pd.read_csv(R / f"lexicon_ws_scaling_{fam}.csv")
        un = un[(un.category == "noun") & (un.kind == "model")].groupby("scale").spearman.mean()
        print(f"  NOTE figS_lexicon_alignment A ({key}): aligned@100k {al.loc[100000]:.3f} vs "
              f"unfiltered@100k {un.loc[100000]:.3f}, @300k {un.loc[300000]:.3f}, "
              f"@1M {un.loc[1000000]:.3f}")
ax.axvline(170000, color=T.SUB, lw=0.6, ls=(0, (1, 2)), zorder=1)
ax.text(170000 / 1.15, -0.045, "all referential pairs", fontsize=5.0, color=T.SUB,
        ha="right", va="bottom")
ax.axhline(0, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)
T.end_labels(ax, [t[0] * 1.15 for t in ends], [t[1] for t in ends], [t[2] for t in ends],
             [t[3] for t in ends], gap=0.02, xl=[t[0] * 1.45 for t in ends], fontsize=5.2)
ax.text(0.04, 0.95, "aligned only (solid)\nunfiltered (dashed, open)", transform=ax.transAxes,
        fontsize=5.4, color=T.INK, va="top", linespacing=1.4)
ax.set_xscale("log"); ax.set_xlim(7e3, 4.5e6); ax.set_ylim(-0.06, 0.35)
ax.set_xlabel("training pairs")
ax.set_ylabel("ρ with human relatedness (noun–noun)")
T.clean(ax)

T.save(fig, "figS_lexicon_alignment")
