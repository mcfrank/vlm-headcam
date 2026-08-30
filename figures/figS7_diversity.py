"""S display item — does it matter WHOSE data it is, holding the amount fixed?

The B26 diversity sweeps: at each pair budget, the pairs are drawn from k children (k random
per seed). One line per budget. Largely a null: at 30k any child's data is as good as any
mix; a modest effect opens at 100k-300k, where a few children's experience is exhausted
before the budget is. Budgets missing small k (300k from 1 or 3 children) are impossible —
those children don't hold that many pairs.
"""
import sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

fams = {}
for f in D.runs.family.unique():
    if m := re.fullmatch(r"B26_div(\w*?)_?(\d+)c", str(f)):
        fams.setdefault(m.group(1) or "30k", []).append((int(m.group(2)), f))
SHADE = {"30k": "#9ec9b8", "100k": "#4d9971", "300k": T.FREE}
fig, ax = plt.subplots(figsize=(T.W1, 2.4))
for budget in ["30k", "100k", "300k"]:
    pts = sorted(fams.get(budget, []))
    ks = [k for k, _ in pts]; fm = [D.family(f) for _, f in pts]
    ax.errorbar(ks, [f["mean"] for f in fm], yerr=[f["sd"] for f in fm], fmt="-o",
                color=SHADE[budget], ms=2.8, lw=1.0, elinewidth=0.6, capsize=1.6, zorder=3)
    ax.text(ks[-1] * 1.22, fm[-1]["mean"], f"{budget}\npairs", fontsize=5.4,
            color=SHADE[budget], va="center", linespacing=1.2)
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
ax.text(1, 25.8, "chance", fontsize=5.4, color=T.SUB, ha="left")
ax.set_xscale("log")
ax.set_xticks([1, 3, 10, 25, 50]); ax.set_xticklabels([1, 3, 10, 25, 50]); ax.minorticks_off()
ax.set_xlim(0.8, 160); ax.set_ylim(20, 70)
ax.set_xlabel("children contributing the pairs")
ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "figS7_diversity")
