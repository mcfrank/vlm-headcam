"""C7 — does a temporal window (±5 s of frames) around the utterance buy anything over the
single midpoint frame?

The window arm replaces the midpoint frame's 4x4 region grid with the grids of all frames
within ±5 s (11 frames at 1 fps; windows were complete, 10.96/11 frames on average), so the
MIL max ranges over time as well as space. Everything else is fixed and each run is paired
seed-for-seed to the region-MIL run on the same manifest, at four scales and for every
encoder.

A: the two arms against each other.
B: the paired difference, seed-matched. The off-the-shelf encoders gain from the window,
   most at full scale; the BabyView-trained encoders never do.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

ENC = [(E["tag"], E["label"], E["color"]) for E in reversed(T.ENCODERS)]   # smallest first
SCALES = [("rand_30000", "win5_30000", "30k"), ("rand_300000", "win5_300000", "300k"),
          ("rand_1000000", "win5_1000000", "1M"), ("base", "win5_full", "1.69M")]
MK = {"30k": "o", "300k": "s", "1M": "^", "1.69M": "D"}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.32))
for enc, key, col in ENC:
    for mid_f, win_f, slab in SCALES:
        mid = D.family(f"F_{enc}_{mid_f}"); win = D.family(f"F_{enc}_{win_f}")
        if win["n"] == 0:
            continue
        ax.errorbar([mid["mean"]], [win["mean"]], xerr=[mid["sd"]], yerr=[win["sd"]],
                    fmt=MK[slab], color=col, ms=3.4, elinewidth=0.6, capsize=0, zorder=3)
lim = [22, 92]
ax.plot(lim, lim, color=T.SUB, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(70, 73.5, "identity", fontsize=5.4, color=T.SUB, ha="center", rotation=41,
        rotation_mode="anchor")
for i, (enc, key, col) in enumerate(ENC):
    ax.scatter([24.5], [89 - i * 3.2], s=14, color=col, zorder=4)
    ax.text(26.5, 89 - i * 3.2, key, fontsize=5.2, color=col, va="center")
for j, slab in enumerate(["30k", "300k", "1M", "1.69M"]):
    ax.scatter([46], [89 - j * 3.2], s=13, facecolors="none", edgecolors=T.SUB, lw=0.8,
               marker=MK[slab], zorder=4)
    ax.text(48, 89 - j * 3.2, slab, fontsize=5.2, color=T.SUB, va="center")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("midpoint frame 4AFC (%)"); ax.set_ylabel("±5 s window 4AFC (%)")
T.clean(ax, grid_axis=None)

# ---- B: paired per-seed differences ---------------------------------------------
for i, (enc, key, col) in enumerate(ENC):
    for j, (mid_f, win_f, slab) in enumerate(SCALES):
        a = D.runs[D.runs.family == f"F_{enc}_{mid_f}"].set_index("seed").best_acc
        b = D.runs[D.runs.family == f"F_{enc}_{win_f}"].set_index("seed").best_acc
        common = sorted(set(a.index) & set(b.index))
        x = i + (j - 1.5) * 0.19
        if not common:
            if j == 0:
                bx.text(i, 0.4, "runs\npending", fontsize=4.6, color=T.SUB, ha="center",
                        va="bottom", linespacing=1.2)
            print(f"  NOTE figS_window {key} {slab}: no window runs yet")
            continue
        d = np.array([b[s] - a[s] for s in common])
        bx.errorbar([x], [d.mean()], yerr=[d.std(ddof=1) / np.sqrt(len(d))], fmt=MK[slab],
                    color=col, ms=2.8, elinewidth=0.7, capsize=1.6, zorder=3)
        bx.scatter([x] * len(d), d, s=4, color=col, alpha=0.35, zorder=2)
        print(f"  NOTE figS_window {key} {slab}: midpoint {a.mean():.1f} vs window "
              f"{b.mean():.1f}, paired Δ {d.mean():+.2f} ± {d.std(ddof=1):.2f} (n={len(d)})")
bx.axhline(0, color=T.SUB, lw=0.7, ls=(0, (4, 3)))
bx.set_xticks(range(len(ENC)))
bx.set_xticklabels([e[1] for e in ENC], fontsize=5.2, rotation=30, ha="right",
                   rotation_mode="anchor")
bx.set_ylim(-8, 8); bx.set_ylabel("window gain (Δ 4AFC, paired)")
bx.text(0.02, 0.97, "30k · 300k · 1M · 1.69M pairs (left to right)", transform=bx.transAxes,
        fontsize=5.4, color=T.SUB, va="top")
T.clean(bx)
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.15)
T.save(fig, "figS_window")
