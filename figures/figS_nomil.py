"""C5 — does region-MIL buy anything over a single mean-pooled frame vector?

The no-MIL arm replaces the 4x4 region grid with its mean (R=1 caches, train and eval),
holding everything else fixed and pairing seed-for-seed against the region-MIL runs on the
same manifests, at three scales and for all six encoders.

A: the two arms against each other (mean-pooled on x, so above the line = MIL gain).
B: the paired difference, seed-matched. Every cell sits within seed noise of zero, so a
   single mean-pooled frame vector learns as well as region-MIL everywhere we tested.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

ENC = [(E["tag"], E["label"], E["color"]) for E in T.ENCODERS]
SCALES = [("30000", "rand_30000", "wf30000", "30k"),
          ("300000", "rand_300000", "wf300000", "300k"),
          ("full", "base", "wffull", "1.69M")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.32))
MK = {"30k": "o", "300k": "s", "1.69M": "D"}
for enc, key, col in ENC:
    for _, mil_f, wf_f, slab in SCALES:
        mil = D.family(f"F_{enc}_{mil_f}"); wf = D.family(f"F_{enc}_{wf_f}")
        ax.errorbar([wf["mean"]], [mil["mean"]], xerr=[wf["sd"]], yerr=[mil["sd"]],
                    fmt=MK[slab], color=col, ms=3.4, elinewidth=0.6, capsize=0, zorder=3)
lim = [22, 90]
ax.plot(lim, lim, color=T.SUB, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(70, 73.5, "identity", fontsize=5.4, color=T.SUB, ha="center", rotation=41,
        rotation_mode="anchor")
for i, (enc, key, col) in enumerate(ENC):
    ax.scatter([24.5], [87 - i * 3.2], s=14, color=col, zorder=4)
    ax.text(26.5, 87 - i * 3.2, key, fontsize=5.2, color=col, va="center")
for j, slab in enumerate(["30k", "300k", "1.69M"]):
    ax.scatter([46], [87 - j * 3.2], s=13, facecolors="none", edgecolors=T.SUB, lw=0.8,
               marker=MK[slab], zorder=4)
    ax.text(48, 87 - j * 3.2, slab, fontsize=5.2, color=T.SUB, va="center")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("mean-pooled frame 4AFC (%)"); ax.set_ylabel("region-MIL 4AFC (%)")
T.clean(ax, grid_axis=None)

# ---- B: paired per-seed differences ---------------------------------------------
for i, (enc, key, col) in enumerate(ENC):
    for j, (_, mil_f, wf_f, slab) in enumerate(SCALES):
        a = D.runs[D.runs.family == f"F_{enc}_{mil_f}"].set_index("seed").best_acc
        b = D.runs[D.runs.family == f"F_{enc}_{wf_f}"].set_index("seed").best_acc
        common = sorted(set(a.index) & set(b.index))
        d = np.array([a[s] - b[s] for s in common])
        x = i + (j - 1) * 0.22
        bx.errorbar([x], [d.mean()], yerr=[d.std(ddof=1) / np.sqrt(len(d))], fmt=MK[slab],
                    color=col, ms=3.0, elinewidth=0.7, capsize=1.6, zorder=3)
        bx.scatter([x] * len(d), d, s=4, color=col, alpha=0.35, zorder=2)
        print(f"  NOTE figS_nomil {key} {slab}: MIL {a.mean():.1f} vs no-MIL {b.mean():.1f}, "
              f"paired Δ {d.mean():+.2f} ± {d.std(ddof=1):.2f} (n={len(d)})")
bx.axhline(0, color=T.SUB, lw=0.7, ls=(0, (4, 3)))
bx.set_xticks(range(len(ENC)))
bx.set_xticklabels([e[1] for e in ENC], fontsize=5.2, rotation=30, ha="right",
                   rotation_mode="anchor")
bx.set_ylim(-7, 7); bx.set_ylabel("region-MIL gain (Δ 4AFC, paired)")
bx.text(0.02, 0.97, "30k · 300k · 1.69M pairs (left to right)", transform=bx.transAxes,
        fontsize=5.4, color=T.SUB, va="top")
T.clean(bx)
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.15)
T.save(fig, "figS_nomil")
