"""Encoder-ladder comparison figures (chapter: how much does the vision encoder matter).
Two candidate designs over the same 3-seed extended ladder: (a) three faceted waterfalls with a
shared y-axis, (b) a slope chart. Ordered worst->best: DINOv3-BV, DINOv2-OTS, DINOv3-OTS."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data2/mcfrank/vlm-headcam/book_figs"
FREE, ORACLE, INK, SUB, GRID = "#1d9e75", "#185fa5", "#2c2c2a", "#6b6a66", "#e1e0d9"
RUNGS = ["whole-frame\n(meanpatch)", "+ region\nMIL", "+ alignment\nfilter", "+ referent\nlabel"]
KIND = [FREE, FREE, ORACLE, ORACLE]
DATA = [  # (label, color, values)  ordered worst -> best
    ("DINOv3-L\n(BabyView)", [41.0, 41.7, 45.7, 51.1]),
    ("DINOv2\n(off-the-shelf)", [61.3, 65.3, 70.3, 81.5]),
    ("DINOv3-B\n(off-the-shelf)", [70.8, 72.6, 75.4, 84.6]),
]
LINE = ["#b0655a", "#6b6a66", "#1d9e75"]
FLOOR = 38

# ---- (a) faceted waterfalls, shared y ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), dpi=150, sharey=True)
for ax, (name, vals) in zip(axes, DATA):
    prev = None
    for i, (v, c) in enumerate(zip(vals, KIND)):
        if i == 0:
            ax.bar(i, v - FLOOR, bottom=FLOOR, width=0.64, color=c, zorder=3)
        else:
            bot, top = min(prev, v), max(prev, v)
            ax.bar(i, top - bot, bottom=bot, width=0.64, color=c, zorder=3)
            ax.text(i, top + 0.7, f"+{v - prev:.1f}", ha="center", fontsize=8.5, color=INK, zorder=4)
            ax.plot([i - 1 + 0.32, i - 0.32], [prev, prev], color=SUB, lw=0.8, ls=(0, (3, 3)), zorder=2)
        ax.text(i, v + (2.4 if i == 0 else -3.2), f"{v:.0f}", ha="center", fontsize=9.5,
                color=INK if i == 0 else "white", fontweight="bold", zorder=5)
        prev = v
    gap = vals[-1] - vals[0]
    ax.annotate("", (2.55, vals[0]), (2.55, vals[-1]), arrowprops=dict(arrowstyle="<->", color=SUB, lw=1.1))
    ax.text(2.68, (vals[0] + vals[-1]) / 2, f"referential\nheadroom\n+{gap:.0f}", fontsize=8.3,
            color=SUB, va="center")
    ax.set_title(name, fontsize=10.5, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(RUNGS, fontsize=7.6)
    ax.set_xlim(-0.6, 3.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SUB, length=0); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylim(FLOOR, 90); axes[0].set_ylabel("Konkle 4AFC", fontsize=10, color=SUB)
fig.suptitle("The bootstrapping ladder across vision encoders (green = free, blue = oracle)",
             fontsize=12.5, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/fig_encoder_ladder_waterfall.png", bbox_inches="tight"); print("wrote waterfall")

# ---- (c) encoder-comparison bar: whole-frame vs region-MIL, all encoders ----
ENC = [  # (label, wf, region)  sorted by whole-frame
    ("V-JEPA2-L\nBabyView", 30.8, 31.4),
    ("ZWM-170M\nBabyView", 31.6, 28.9),
    ("ZWM-1B\nBabyView", 35.6, 35.7),
    ("DINOv3-L\nBabyView", 41.0, 41.7),
    ("DINOv2\noff-the-shelf", 61.3, 65.3),
    ("DINOv3-B\noff-the-shelf", 70.8, 72.6),
]
fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=150)
x = range(len(ENC)); w = 0.38
ax.bar([i - w / 2 for i in x], [e[1] for e in ENC], w, color="#9ec9b8", label="whole-frame", zorder=3)
ax.bar([i + w / 2 for i in x], [e[2] for e in ENC], w, color=FREE, label="+ region-MIL", zorder=3)
for i, e in enumerate(ENC):
    ax.text(i - w / 2, e[1] + 0.8, f"{e[1]:.0f}", ha="center", fontsize=8, color=SUB)
    ax.text(i + w / 2, e[2] + 0.8, f"{e[2]:.0f}", ha="center", fontsize=8, color=INK)
ax.axhline(25, color=SUB, lw=1, ls=(0, (5, 4))); ax.text(5.4, 26, "chance", color=SUB, fontsize=8, ha="right")
ax.set_xticks(list(x)); ax.set_xticklabels([e[0] for e in ENC], fontsize=8.3)
ax.set_ylim(0, 82); ax.set_ylabel("Konkle 4AFC", fontsize=10, color=SUB)
ax.set_title("Off-the-shelf encoders dominate; BabyView-trained encoders transfer worse",
             fontsize=11.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=SUB, length=0); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_encoder_compare.png", bbox_inches="tight"); print("wrote compare")

# ---- (b) slope chart ----
fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
for (name, vals), col in zip(DATA, LINE):
    ax.plot(range(4), vals, "-o", color=col, lw=2, ms=6, zorder=3)
    ax.text(3.05, vals[-1], f"  {name.split(chr(10))[0]} ({name.split(chr(10))[1]})".replace("(", "").replace(")", ""),
            fontsize=8.8, color=col, va="center")
    ax.annotate(f"+{vals[-1]-vals[0]:.0f}", (0, vals[0]), (-0.5, vals[0]), fontsize=8.5, color=col, va="center", ha="right")
ax.axvspan(-0.4, 1.4, color=FREE, alpha=0.05); ax.axvspan(1.6, 3.4, color=ORACLE, alpha=0.05)
ax.text(0.5, 39, "free", color=FREE, fontsize=9, ha="center"); ax.text(2.5, 39, "oracle", color=ORACLE, fontsize=9, ha="center")
ax.set_xticks(range(4)); ax.set_xticklabels([r.replace("\n", " ") for r in RUNGS], fontsize=9)
ax.set_ylim(38, 90); ax.set_xlim(-0.9, 4.2); ax.set_ylabel("Konkle 4AFC", fontsize=10, color=SUB)
ax.set_title("Referential headroom (base → label ceiling) shrinks with a stronger encoder — but persists",
             fontsize=11, color=INK, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=SUB, length=0); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_encoder_ladder_slope.png", bbox_inches="tight"); print("wrote slope")
