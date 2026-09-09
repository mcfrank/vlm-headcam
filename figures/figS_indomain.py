"""the eval is out-of-domain; does that flatter the off-the-shelf encoders?

Konkle photos are objects on white backgrounds, so a reviewer can reasonably ask whether
the off-the-shelf advantage is a domain match rather than a representational one. Here the
same six models are evaluated on HELD-OUT BabyView frames — 603 frames, 82 concrete-object
categories, 356 videos never trained on, one frame per video per category, near-duplicate
frames dropped — i.e. the BabyView-trained encoders' own domain.

A: in-domain word-learning 4AFC across scale, the same construction as fig2A.
B: the off-the-shelf advantage (mean OTS minus mean BabyView-trained) on each eval. It
   shrinks in domain but does not close, so a slice of the Konkle gap is domain match and
   the rest is not.
"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
ind = pd.read_csv(R / "indomain_eval.csv")
ENC = [(E["tag"], E["label"], E["color"], E["marker"]) for E in T.ENCODERS]
OTS = [E["tag"] for E in T.ENCODERS if E["regime"] == "OTS"]
BV = [E["tag"] for E in T.ENCODERS if E["regime"] == "BV"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.30))

# ---- A: in-domain scaling --------------------------------------------------------
ends = []
for enc, key, col, mk in ENC:
    g = ind[ind.encoder == enc].groupby("N").acc.agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt="-" + mk, color=col, ms=2.6, lw=1.1,
                elinewidth=0.6, capsize=1.4, zorder=3)
    ends.append((g.index.max(), g["mean"].iloc[-1], key, col))
T.end_labels(ax, [t[0] * 1.25 for t in ends], [t[1] for t in ends], [t[2] for t in ends],
             [t[3] for t in ends], gap=3.2, xl=[t[0] * 1.6 for t in ends])
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(3.0e6, 22.4, "chance", fontsize=5.4, color=T.SUB, ha="right")
ax.set_xscale("log"); ax.set_xlim(2e3, 5e6); ax.set_ylim(18, 82)
ax.set_xlabel("training pairs")
ax.set_ylabel("in-domain 4AFC (%)\nheld-out BabyView frames")
T.clean(ax)

# ---- B: the off-the-shelf advantage on each eval ---------------------------------
kon = {}
for e, _, _, _ in ENC:
    for f in D.runs.family.unique():
        if m := re.fullmatch(rf"F_{e}_rand_(\d+)", str(f)):
            kon[(e, int(m.group(1)))] = D.family(f)["mean"]
    kon[(e, 1686105)] = D.family(f"F_{e}_base")["mean"]
Ns = sorted(ind.N.unique())
rows = []
for N in Ns:
    io = ind[ind.N == N].groupby("encoder").acc.mean()
    rows.append((N,
                 np.mean([kon[(e, N)] for e in OTS]) - np.mean([kon[(e, N)] for e in BV]),
                 np.mean([io[e] for e in OTS]) - np.mean([io[e] for e in BV])))
g = pd.DataFrame(rows, columns=["N", "konkle", "indomain"])
for c, lab, col, ls in [("konkle", "Konkle (out-of-domain)", T.INK, "-"),
                        ("indomain", "held-out BabyView frames", T.ORACLE, (0, (2.5, 1.5)))]:
    bx.plot(g.N, g[c], ls=ls, marker="o", ms=2.8, lw=1.1, color=col, zorder=3)
    bx.text(g.N.iloc[-1] * 1.25, g[c].iloc[-1], lab.split(" (")[0], fontsize=5.4, color=col,
            va="center")
bx.axhline(0, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.set_xscale("log"); bx.set_xlim(2e3, 2.2e7); bx.set_ylim(-3, 44)
bx.set_xlabel("training pairs")
bx.set_ylabel("off-the-shelf advantage\n(mean OTS − mean BabyView-trained, pts)")
T.clean(bx)
print(f"  NOTE figS_indomain: {ind.n_cats.iloc[0]} categories, {int(ind.n_oov.max())} out of "
      f"model vocabulary at the smallest scale; full-corpus in-domain "
      + ", ".join(f"{k}={ind[(ind.encoder==e)&(ind.N==1686105)].acc.mean():.1f}"
                  for e, k, _, _ in ENC)
      + f"; advantage at full corpus: Konkle {g.konkle.iloc[-1]:.1f} vs in-domain "
        f"{g.indomain.iloc[-1]:.1f} pts")
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.16)
T.save(fig, "figS_indomain")
