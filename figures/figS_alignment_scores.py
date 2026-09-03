"""Distribution of Gemini referential-alignment ratings.

Every (utterance, frame) pair in the 2026.1 release was rated 0-100 for how strongly the
utterance refers to a concrete object visible in the frame (prompt in the SI text).

A: the rating distribution over all scored pairs. Counts are on a log axis because the
   distribution is extremely bimodal -- 89.9% of pairs are rated exactly 0 -- and the
   aligned mass would otherwise be invisible. Note the aligned mass is spread over 50-100
   rather than piled at 100.
B: the same selection expressed per child, to show that the ~10% aligned fraction is a
   property of the corpus rather than of a few children. Intervals are a cluster bootstrap
   over that child's VIDEOS, not over pairs: pairs within a video share a scene and a
   speaker, so treating them as independent understates the uncertainty about two-fold.

Source: the committed release diagnostics, not a live cluster read.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

D = __import__("pathlib").Path(__file__).resolve().parent.parent / "diagnostics" / "2026.1"
d = pd.read_parquet(D / "dist_alignment.parquet").sort_values("alignment_bin")
V = pd.read_parquet(D / "video_level.parquet")
V = V[V.ref_scored > 0]
rng = np.random.default_rng(0)
rows = []
for ch, g in V.groupby("child"):
    h, n = g.ref_has_referent.values, g.ref_scored.values
    i = rng.integers(0, len(h), (2000, len(h)))          # resample videos, not pairs
    bs = 100 * h[i].sum(1) / n[i].sum(1)
    rows.append(dict(child=ch, rate=100 * h.sum() / n.sum(), n_videos=len(g),
                     lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
C = pd.DataFrame(rows)

tot = d["count"].sum()
ge = d[d.alignment_bin >= 50]["count"].sum()
zero = int(d[d.alignment_bin == 0]["count"].iloc[0])
mid = int(d[(d.alignment_bin > 0) & (d.alignment_bin < 50)]["count"].sum())

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.5),
                             gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.30))

# ---- A: the rating distribution --------------------------------------------------
col = [T.NEUTRAL if b < 50 else T.ORACLE for b in d.alignment_bin]
ax.bar(d.alignment_bin, d["count"], width=7, color=col, zorder=3)
ax.set_yscale("log"); ax.set_ylim(0.5, 5e6)
ax.set_xlim(-8, 108); ax.set_xticks(range(0, 101, 20))
ax.axvline(45, color=T.SUB, lw=0.7, ls=(0, (3, 2)), zorder=2)
ax.text(47, 2.2e6, "kept as\naligned", fontsize=5.4, color=T.ORACLE, va="top", linespacing=1.3)
ax.text(0, zero * 1.6, f"{100 * zero / tot:.1f}%", fontsize=5.6, color=T.SUB, ha="center")
ax.text(75, ge * 2.4, f"{100 * ge / tot:.1f}% of pairs\n({ge:,})", fontsize=5.6,
        color=T.ORACLE, ha="center", linespacing=1.3)
ax.set_xlabel("Gemini referential-alignment rating")
ax.set_ylabel("utterance–frame pairs (log)")
T.clean(ax)

# ---- B: per child ----------------------------------------------------------------
o = C.sort_values("rate").reset_index(drop=True)
bx.errorbar(range(len(o)), o.rate, yerr=[o.rate - o.lo, o.hi - o.rate], fmt="o",
            color=T.ORACLE, ms=2.6, elinewidth=0.6, capsize=0, zorder=3)
bx.axhline(100 * ge / tot, color=T.SUB, lw=0.7, ls=(0, (3, 2)), zorder=2)
bx.text(len(o) - 0.5, 100 * ge / tot + 0.6, "corpus", fontsize=5.4, color=T.SUB, ha="right")
bx.set_xlabel(f"child, ordered by rate (n={len(o)})")
bx.set_xlim(-1.5, len(o) + 0.5)
bx.set_ylabel("pairs with a referent (%)")
bx.set_ylim(0, 27)
T.clean(bx)

print(f"  NOTE figS_alignment_scores: {tot:,} scored pairs; {100*zero/tot:.1f}% exactly 0, "
      f"{mid:,} between 0 and 50, {100*ge/tot:.1f}% >= 50. Per-child referent rate median "
      f"{o.rate.median():.1f}% (IQR {o.rate.quantile(.25):.1f}-{o.rate.quantile(.75):.1f}, "
      f"range {o.rate.min():.1f}-{o.rate.max():.1f}); video-cluster bootstrap, "
      f"{o.n_videos.min()}-{o.n_videos.max()} videos per child, "
      f"{int((o.n_videos < 5).sum())} children with <5")
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.15)
T.save(fig, "figS_alignment_scores")
