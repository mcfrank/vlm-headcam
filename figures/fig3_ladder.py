"""Display item 3 — the ladder at three scales: what oracle information buys, and how it
shrinks as raw experience grows.

Each scale is a matched subsample of the 2026.1 corpus trained four ways (same model, same
eval): base = every pair as spoken; then three oracle rungs answering which moments are
referential (alignment filter), which word is the referent (word selection), and which object
it names (vision binding — text is the label itself, the clean-label ceiling). x is the raw
experience of the base arm; the oracle rungs keep only ~10% of it. The shaded wedge between
the free line and the ceiling is the referential headroom, closing with scale.

Read from runs.parquet directly (published.csv claim ids still point at the 2025.2 ladder).
"""
import sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
import numpy as np

RUNGS = [("base", "unfiltered (free)", None),
         ("filtnat", "+ alignment filter", "which moments"),
         ("t15", "+ word selection", "which word"),
         ("t2", "+ vision binding", "which object")]

# scales present in the scrape: B26_lad<scale>_<rung>, empty scale = full corpus
scales = sorted({m.group(1) for f in D.runs.family.unique()
                 if (m := re.fullmatch(r"B26_lad(\d*)_?(?:base|filtnat|t15|t2)", str(f)))},
                key=lambda s: int(s or 10**9))
fam = lambda sc, r: D.family(f"B26_lad{sc}_{r}" if sc else f"B26_lad_{r}")
X = np.array([fam(sc, "base")["n_pairs"] for sc in scales])
L = {r: dict(y=np.array([fam(sc, r)["mean"] for sc in scales]),
             e=np.array([fam(sc, r)["sd"] for sc in scales])) for r, _, _ in RUNGS}

fig, ax = plt.subplots(figsize=(T.W15, 2.7))

ax.fill_between(X, L["base"]["y"], L["t2"]["y"], color=T.ORACLE, alpha=0.07, lw=0, zorder=1)
STYLE = {"base": dict(color=T.FREE, ls="-", marker="o", lw=1.2),
         "filtnat": dict(color=T.ORACLE, ls="-", marker="s", lw=1.0, alpha=0.75),
         "t15": dict(color=T.ORACLE, ls=(0, (2, 1.5)), marker="^", lw=1.0, alpha=0.75),
         "t2": dict(color=T.ORACLE, ls="-", marker="D", lw=1.4)}
for r, lab, q in RUNGS:
    st = STYLE[r]
    ax.errorbar(X, L[r]["y"], yerr=L[r]["e"], ms=3.0, elinewidth=0.7, capsize=1.6, zorder=3, **st)

# the headroom at each scale, above the ceiling point where nothing else lives
for i, sc in enumerate(scales):
    d = L["t2"]["y"][i] - L["base"]["y"][i]
    lab = f"headroom +{d:.1f}" if i == 0 else f"+{d:.1f}"
    ax.text(X[i], L["t2"]["y"][i] + L["t2"]["e"][i] + 1.2, lab, fontsize=5.6, color=T.SUB,
            ha="center", va="bottom")

# direct labels at the right edge, spread apart, tied to their lines by thin leaders
ends = {r: L[r]["y"][-1] for r, _, _ in RUNGS}
ANCHOR = {"t2": 87.6, "filtnat": 84.4, "t15": 81.0, "base": 77.6}
for r, lab, q in RUNGS:
    col = T.FREE if r == "base" else T.ORACLE
    a = STYLE[r].get("alpha", 1.0)
    ax.plot([X[-1] * 1.06, X[-1] * 1.20], [ends[r], ANCHOR[r]], color=col, lw=0.5,
            alpha=0.6 * a, zorder=2)
    ax.text(X[-1] * 1.25, ANCHOR[r], lab if q is None else f"{lab}  ({q})", fontsize=5.6,
            color=col, va="center", alpha=a)

ax.set_xscale("log")
ax.set_xlim(6.5e4, 1.6e7)
ax.set_xticks(X)
ax.set_xticklabels([f"{x/1e3:.0f}k" if x < 1e6 else f"{x/1e6:.2f}M\n(full corpus)" for x in X],
                   fontsize=6.2)
ax.set_ylim(42, 90)
ax.set_xlabel("raw experience (pairs; oracle rungs keep ~10% of it)")
ax.set_ylabel("Konkle 4AFC (%)")
ax.minorticks_off()
T.clean(ax)
print("  NOTE fig3: ladder at scales", [int(x) for x in X], "· n=3 per point ·",
      "word selection inverts at full scale (82.8 < 84.6)")
T.save(fig, "fig3_ladder")
