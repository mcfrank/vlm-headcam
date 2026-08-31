"""S display item — does it matter WHOSE data it is, holding the amount fixed?

The FINAL-corpus diversity sweeps (DINOv3-L encoder): at each pair budget, the pairs are drawn from k children (k random
per seed). One line per budget. A null on the final rig: at both budgets, accuracy is flat in the
number of contributing children. Budgets missing small k are impossible —
those children don't hold that many pairs.
"""
import sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

fams = {}
for f in D.runs.family.unique():
    if m := re.fullmatch(r"F_dinov3l_div(\w+?)_(\d+)c", str(f)):
        fams.setdefault(m.group(1), []).append((int(m.group(2)), f))
SHADE = {"100k": "#4d9971", "300k": T.FREE}
fig, ax = plt.subplots(figsize=(T.W1, 2.4))
for budget in ["100k", "300k"]:
    pts = sorted(fams.get(budget, []))
    ks = [k for k, _ in pts]; fm = [D.family(f) for _, f in pts]
    ax.errorbar(ks, [f["mean"] for f in fm], yerr=[f["sd"] for f in fm], fmt="-o",
                color=SHADE[budget], ms=2.8, lw=1.0, elinewidth=0.6, capsize=1.6, zorder=3)
    ax.text(ks[-1] * 1.22, fm[-1]["mean"], f"{budget}\npairs", fontsize=5.4,
            color=SHADE[budget], va="center", linespacing=1.2)
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
ax.text(1, 25.8, "chance", fontsize=5.4, color=T.SUB, ha="left")
ax.set_xscale("log")
ax.set_xticks([1, 3, 10, 25, 48]); ax.set_xticklabels([1, 3, 10, 25, 48]); ax.minorticks_off()
ax.set_xlim(0.8, 160); ax.set_ylim(20, 70)
ax.set_xlabel("children contributing the pairs")
ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "figS7_diversity")
