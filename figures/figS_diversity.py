"""S display item — does it matter WHOSE data it is, holding the amount fixed?

The FINAL-corpus diversity sweeps (DINOv3-L encoder): at each pair budget the pairs are drawn
from k children, redrawn per seed. One line per budget, a null at all three: accuracy is flat
in the number of contributing children wherever the comparison is possible. Cells missing at
small k are impossible rather than absent — those children do not hold that many pairs. Open
markers are cells whose seeds are still landing.
"""
import sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

fams = {}
for f in D.runs.family.unique():
    if m := re.fullmatch(r"F_dinov3l_div(\w+?)_(\d+)c", str(f)):
        fams.setdefault(m.group(1), []).append((int(m.group(2)), f))
SHADE = {"30k": "#9ecae1", "100k": T.OTHER, "300k": T.FREE}
fig, ax = plt.subplots(figsize=(T.W1, 2.4))
for budget in ["30k", "100k", "300k"]:
    pts = sorted(fams.get(budget, []))
    ks = [k for k, _ in pts]; fm = [D.family(f) for _, f in pts]
    nmax = max(f["n"] for f in fm)
    ax.plot(ks, [f["mean"] for f in fm], "-", color=SHADE[budget], lw=1.0, zorder=3)
    for k_, f in zip(ks, fm):                  # open marker where seeds are still landing
        ax.errorbar([k_], [f["mean"]], yerr=[f["sd"]], fmt="o", color=SHADE[budget], ms=2.8,
                    elinewidth=0.6, capsize=1.6, zorder=4,
                    markerfacecolor=SHADE[budget] if f["n"] == nmax else "white",
                    markeredgewidth=0.8)
    ax.text(ks[-1] * 1.22, fm[-1]["mean"], f"{budget}\npairs", fontsize=5.4,
            color=SHADE[budget], va="center", linespacing=1.2)
    short = [f"{k}c:n={f['n']}" for k, f in zip(ks, fm) if f["n"] < nmax]
    print(f"  NOTE figS_diversity {budget}: k={ks}, n={[f['n'] for f in fm]}"
          + (f"  UNDER-SEEDED {short}" if short else ""))
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
ax.text(1, 25.8, "chance", fontsize=5.4, color=T.SUB, ha="left")
ax.set_xscale("log")
ax.set_xticks([1, 3, 10, 25, 48]); ax.set_xticklabels([1, 3, 10, 25, 48]); ax.minorticks_off()
ax.set_xlim(0.8, 160); ax.set_ylim(25, 72)
ax.set_xlabel("children contributing the pairs")
ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "figS_diversity")
