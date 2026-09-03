"""C3 — does training on the child's own speech matter?

Child utterances (KCHI) are 22.9% of the corpus. Removing them leaves 1,299,191 pairs; the
control is a random draw of the same size from the full corpus, so the comparison is
amount-matched and only the speaker composition differs. L-OTS, 3 seeds each.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

ARMS = [("F_dinov3l_randmatch", "matched random draw\n(same N, all speakers)", T.NEUTRAL),
        ("F_dinov3l_nokchi", "child speech removed\n(same N, no KCHI)", T.OTHER)]
fig, ax = plt.subplots(figsize=(T.W1, 2.2))
for i, (fam, lab, col) in enumerate(ARMS):
    f = D.family(fam)
    ax.bar(i, f["mean"] - 25, bottom=25, width=0.5, color=col, zorder=3, yerr=f["sd"],
           error_kw=dict(elinewidth=0.7, capsize=2.2, ecolor=T.INK))
    ax.text(i, f["mean"] + f["sd"] + 1.0, f"{f['mean']:.1f}", ha="center", fontsize=6,
            color=T.INK)
    ax.scatter([i] * f["n"], f["values"], s=5, color=T.INK, alpha=0.5, zorder=4)
    print(f"  NOTE figS_kchi {fam}: {f['mean']:.2f} ± {f['sd']:.2f} (n={f['n']}, "
          f"{f['n_pairs']:,.0f} pairs)")
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
ax.text(1.45, 26, "chance", fontsize=5.4, color=T.SUB, ha="right")
ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([l for _, l, _ in ARMS], fontsize=5.6)
ax.set_xlim(-0.6, 1.6); ax.set_ylim(22, 92)
ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "figS_kchi")
