"""what referential selection does to the LEXICON.

A: matched-data comparison. Human-relatedness rho for the aligned arm (trained only on
   referential pairs) against the unfiltered arm, on the same x-axis of training pairs.
   Alignment buys a far more human-like word space at matched data -- the same
   data-efficiency shape fig2 shows on 4AFC, in an independent measure.
B: the ladder at ~170k pairs. Being told which word or which object on top of which
   moments adds nothing to the lexicon, even though those rungs score better on 4AFC.
   The word2vec control is trained on each rung's OWN text, so it degrades as that text is
   stripped toward bare referent labels -- at the top rung the grounded model overtakes it.

Noun-noun pairs throughout; points are seeds, bars are across-seed SD.
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
RUNGS = ["aligned", "filtnat", "word selection", "vision binding"]
RLAB = ["aligned\nonly", "+ alignment\nfilter", "+ word\nselection", "+ vision\nbinding"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.7),
                             gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.30))

# ---- A: aligned vs unfiltered at matched data -----------------------------------
for enc, fam, key, col in ENC:
    for tag, ls, mk, fc in [("", (0, (2.5, 1.5)), "o", "white"), ("-aligned", "-", "o", None)]:
        d = pd.read_csv(R / f"lexicon_ws_scaling_{fam}{tag}.csv")
        d = d[(d.category == "noun") & (d.kind == "model") & (d.scale >= 1e4)]
        g = d.groupby("scale").spearman.agg(["mean", "std"])
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
ax.text(1.9e6, 0.305, T.enc("dinov3l")["label"], fontsize=5.4, color=T.OTHER, ha="right")
ax.text(0.04, 0.95, "aligned only (solid)\nunfiltered (dashed, open)", transform=ax.transAxes,
        fontsize=5.4, color=T.INK, va="top", linespacing=1.4)
ax.set_xscale("log"); ax.set_xlim(7e3, 3e6); ax.set_ylim(-0.06, 0.35)
ax.set_xlabel("training pairs")
ax.set_ylabel("ρ with human relatedness (noun–noun)")
T.clean(ax)

# ---- B: the ladder at matched pairs ---------------------------------------------
rg = pd.read_csv(R / "lexicon_rungs.csv")
rg = rg[rg.category == "noun"]
xs = np.arange(len(RUNGS))
ends = []
for enc, fam, key, col in ENC:
    d = rg[rg.encoder == enc]
    m = [d[d.rung == r].spearman.mean() for r in RUNGS]
    e = [d[d.rung == r].spearman.std() for r in RUNGS]
    bx.errorbar(xs, m, yerr=e, color=col, marker="o", ms=2.8, lw=1.1, elinewidth=0.6,
                capsize=1.4, zorder=3)
    ends.append((m[-1], key, col))
    if enc == "dinov3l":
        print(f"  NOTE figS_lexicon_alignment B ({key}): " +
              "  ".join(f"{r}={v:.3f}" for r, v in zip(RUNGS, m)))
T.end_labels(bx, [len(RUNGS) - 1] * len(ends), [t[0] for t in ends], [t[1] for t in ends],
             [t[2] for t in ends], gap=0.022, xl=len(RUNGS) - 0.85, fontsize=5.2)
w2v = [rg[rg.rung == r].w2v_spearman.mean() for r in RUNGS]
bx.plot(xs, w2v, "--s", color=T.SUB, ms=2.4, lw=0.9, zorder=2)
bx.text(len(RUNGS) - 0.85, w2v[-1], "word2vec", fontsize=5.2, color=T.SUB, va="center")
print("  NOTE figS_lexicon_alignment B word2vec (own text per rung): " +
      "  ".join(f"{r}={v:.3f}" for r, v in zip(RUNGS, w2v)))
bx.axhline(0, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)
bx.set_xticks(xs); bx.set_xticklabels(RLAB, fontsize=5.6)
bx.set_xlim(-0.4, len(RUNGS) - 0.15); bx.set_ylim(-0.12, 0.40)
bx.set_ylabel("ρ with human relatedness (noun–noun)")
bx.text(0.04, 0.95, "~170k pairs throughout", transform=bx.transAxes, fontsize=5.4,
        color=T.INK, va="top")
T.clean(bx)

for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.15)
T.save(fig, "figS_lexicon_alignment")
