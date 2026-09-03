"""human-relatedness correlations by word class, for every lexicon we have extracted.

fig4C generalized: rho with human relatedness against training scale, for noun-noun pairs
and for all pairs, with the word2vec topline trained on the same utterances and the
two-tower's partial correlation controlling it.

All four final-corpus encoders; the preview families (B26/L-OTS) are ignored. Each family
is scored against the word2vec trained on its OWN corpus. The encoder ordering is the same
one fig2 establishes on 4AFC, which is the point: representation quality propagates from
the eval, through relatedness, to the variance word2vec cannot explain.
"""
import sys, glob, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
ORDER = [("F-dinov3l", "L-OTS", T.OTHER), ("F-dinov3b", "B-OTS", T.FREE),
         ("F-vitb_bv", "B-BV", T.INDOM), ("F-vits_bv", "S-BV", T.INDOM2)]
have = {re.search(r"lexicon_ws_scaling_(.+)\.csv$", f).group(1)
        for f in glob.glob(str(R / "lexicon_ws_scaling_*.csv"))}
ORDER = [o for o in ORDER if o[0] in have]
COL = {f: c for f, _, c in ORDER}
LAB = {f: k for f, k, _ in ORDER}
fams = [f for f, _, _ in ORDER]
print(f"  NOTE figS_lexicon_relatedness: final-corpus lexicons {[LAB[f] for f in fams]}"
      f" (ignoring preview families {sorted(have - set(fams))})")

fig, axes = plt.subplots(1, 2, figsize=(T.W2, 2.5), sharey=True)
for a, (cat, title) in zip(axes, [("noun", "noun–noun pairs"), ("all", "all pairs")]):
    for fam in fams:
        ws = pd.read_csv(R / f"lexicon_ws_scaling_{fam}.csv")
        d = ws[(ws.category == cat) & (ws.scale >= 1e4)]
        col = COL[fam]
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
        a.text(d.scale.max() * 1.25, m["mean"].iloc[-1], LAB[fam], fontsize=5.2,
               color=col, va="center")
        if cat == "noun":
            print(f"     {LAB[fam]}: rho {m['mean'].iloc[-1]:.3f}, "
                  f"w2v {w.values[-1]:.3f}, n_pairs {int(d.n_pairs.max())}")
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
T.save(fig, "figS_lexicon_relatedness")
