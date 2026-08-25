"""Display item 7 — does it matter WHOSE data it is, holding the amount fixed?

The B26 diversity sweeps: at each pair budget, draw the pairs from k children (k random per
seed). One line per budget. At 30k pairs any child's data is as good as any mix; by 100k-300k
a diversity effect opens up — a few children's experience is exhausted before the budget is.
Budgets missing small k (300k from 1 or 3 children) are impossible: those children don't have
that many pairs.
"""
import sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

fams = {}
for f in D.runs.family.unique():
    if m := re.fullmatch(r"B26_div(\w*?)_?(\d+)c", str(f)):
        fams.setdefault(m.group(1) or "30k", []).append((int(m.group(2)), f))
if any(re.match(r"B26_(bigchild|pool)", str(f)) for f in D.runs.family.unique()):
    print("  NOTE fig7: B26 within-child-ceiling families exist — add the matched-count comparison")

SHADE = {"30k": "#9ec9b8", "100k": "#4d9971", "300k": T.FREE}
fig, ax = plt.subplots(figsize=(T.W15, 2.5))
for budget in ["30k", "100k", "300k"]:
    pts = sorted(fams.get(budget, []))
    if not pts:
        continue
    ks = [k for k, _ in pts]; fam = [D.family(f) for _, f in pts]
    col = SHADE[budget]
    ax.errorbar(ks, [f["mean"] for f in fam], yerr=[f["sd"] for f in fam], fmt="-o", color=col,
                ms=3.0, lw=1.1, elinewidth=0.7, capsize=1.8, zorder=3)
    ax.text(ks[-1] * 1.18, fam[-1]["mean"], f"{budget} pairs", fontsize=5.8, color=col,
            va="center")
    print(f"  NOTE fig7 {budget}: k={ks}, n per point =", [f['n'] for f in fam])
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
ax.text(1, 25.8, "chance", fontsize=5.6, color=T.SUB, ha="left")
ax.set_xscale("log")
ax.set_xticks([1, 3, 10, 25, 50]); ax.set_xticklabels([1, 3, 10, 25, 50]); ax.minorticks_off()
ax.set_xlim(0.8, 110)
ax.set_ylim(20, 70)
ax.set_xlabel("children contributing the pairs")
ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "fig7_diversity")
