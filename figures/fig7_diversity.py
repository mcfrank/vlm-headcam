"""Display item 7 — does it matter WHOSE data it is, holding the amount fixed?

The B26 diversity sweep: 30k pairs throughout, drawn from k children (k random per seed),
as a line over k so further budgets can join as lines. The within-child-ceiling
comparison (biggest single child vs pooled at matched count) returns when it is re-run on the
B26 rig — detected below.
"""
import sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

fams = sorted((int(m.group(1)), f) for f in D.runs.family.unique()
              if (m := re.fullmatch(r"B26_div_(\d+)c", str(f))))
if any(re.match(r"B26_(bigchild|pool)", str(f)) for f in D.runs.family.unique()):
    print("  NOTE fig7: B26 within-child-ceiling families exist — add the matched-count panel")

fig, ax = plt.subplots(figsize=(T.W1, 2.2))
ks, fam = [k for k, _ in fams], [D.family(f) for _, f in fams]
ax.errorbar(ks, [f["mean"] for f in fam], yerr=[f["sd"] for f in fam], fmt="-o", color=T.FREE,
            ms=3.2, lw=1.1, elinewidth=0.7, capsize=2, zorder=3)
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
ax.text(50, 25.7, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.set_xscale("log")
ax.set_xticks(ks); ax.set_xticklabels(ks); ax.minorticks_off()
ax.set_xlabel("children contributing (30,000 pairs throughout)")
ax.set_ylabel("Konkle 4AFC (%)")
ax.set_ylim(20, 42)
T.clean(ax)
print("  NOTE fig7: n per bar =", [f["n"] for f in fam], "· seed sd error bars")
T.save(fig, "fig7_diversity")
