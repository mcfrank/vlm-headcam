"""S5 — human-relatedness correlations by word class, for every lexicon we have extracted.

fig4C generalized: rho with human relatedness against training scale, for noun-noun pairs
and for all pairs, with the word2vec topline trained on the same utterances and the
two-tower's partial correlation controlling it.

NB the extracted lexicons are still PREVIEW-corpus runs (C8_*/B26_*), because no lexicon
cache has been built for the final F_ families yet; the build logs which are present. The
figure regenerates across all four final encoders once those caches exist.
"""
import sys, glob, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
COL = {"L-OTS": T.OTHER, "B26": T.FREE, "L-BV": T.INDOM, "B-BV": T.INDOM, "S-BV": T.INDOM2}
LAB = {"B26": "B-OTS (preview)", "L-OTS": "L-OTS (preview)"}
fams = sorted(re.search(r"lexicon_ws_scaling_(.+)\.csv$", f).group(1)
              for f in glob.glob(str(R / "lexicon_ws_scaling_*.csv")))
print(f"  NOTE figS5: lexicons extracted for {fams}; none yet for the final F_ families")

fig, axes = plt.subplots(1, 2, figsize=(T.W2, 2.5), sharey=True)
for a, (cat, title) in zip(axes, [("noun", "noun–noun pairs"), ("all", "all pairs")]):
    for fam in fams:
        ws = pd.read_csv(R / f"lexicon_ws_scaling_{fam}.csv")
        d = ws[(ws.category == cat) & (ws.scale >= 1e4)]
        col = COL.get(fam, T.SUB)
        m = d[d.kind == "model"].groupby("scale").spearman.agg(["mean", "std"])
        a.errorbar(m.index, m["mean"], yerr=m["std"], fmt="-o", color=col, ms=2.6, lw=1.0,
                   elinewidth=0.6, capsize=1.4, zorder=3)
        w = d[d.kind == "w2v"].groupby("scale").spearman.mean()
        a.plot(w.index, w.values, "--s", color=T.SUB, ms=2.2, lw=0.9, zorder=2, alpha=0.8)
        try:
            pt = pd.read_csv(R / f"lexicon_partial_{fam}.csv")
            g = pt[(pt.category == cat) & (pt.scale >= 1e4)].groupby("scale").partial_model
            a.errorbar(g.mean().index, g.mean(), yerr=g.std(), fmt=":o", color=col, ms=2.2,
                       lw=0.9, elinewidth=0.5, capsize=1.2, markerfacecolor="white", zorder=2)
        except FileNotFoundError:
            pass
        a.text(d.scale.max() * 1.25, m["mean"].iloc[-1], LAB.get(fam, fam), fontsize=5.2,
               color=col, va="center")
    a.axhline(0, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)
    a.text(0.03, 0.95, title, transform=a.transAxes, fontsize=6.2, color=T.INK, va="top")
    a.set_xscale("log"); a.set_xlim(6e3, 9e6); a.set_ylim(-0.12, 0.55)
    a.set_xlabel("training pairs")
    T.clean(a)
axes[0].set_ylabel("ρ with human relatedness")
axes[0].text(0.03, 0.86, "dashed grey = word2vec, same utterances\ndotted = word2vec partialled out",
             transform=axes[0].transAxes, fontsize=5.0, color=T.SUB, va="top", linespacing=1.4)
for a, l in zip(axes, "AB"):
    T.panel(a, l, dx=-0.1)
T.save(fig, "figS5_lexicon_encoders")
