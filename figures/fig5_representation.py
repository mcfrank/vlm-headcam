"""Display item 5 — the vision encoder sets the level; referential alignment sets the gap.

A: six encoders, whole-frame vs region-MIL, on identical data (the encoder is the only variable).
B: the ladder re-run under three encoders — the referential headroom shrinks with a stronger
   encoder but never closes.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
import numpy as np

ENC = [("V-JEPA2-L\nBabyView", "enc_vjepa2_wf", "enc_vjepa2_grid"),
       ("ZWM-170M\nBabyView", "enc_zwm170_wf", "enc_zwm170_grid"),
       ("ZWM-1B\nBabyView", "enc_zwm1b_wf", "enc_zwm1b_grid"),
       ("DINOv3-L\nBabyView", "enc_dinov3l_wf", "enc_dinov3l_grid"),
       ("DINOv2\noff-the-shelf", "ladder_pure", "ladder_region"),
       ("DINOv3-B\noff-the-shelf", "enc_dinov3b_wf", "enc_dinov3b_grid")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))

# ---- A ------------------------------------------------------------------------
x = np.arange(len(ENC)); w = 0.37
for off, key, col, lab in [(-w/2, 1, "#9ec9b8", "whole-frame"), (w/2, 2, T.GREEN, "+ region-MIL")]:
    vals = [D.claim(e[key])["value"] for e in ENC]
    prov = [D.claim(e[key])["provisional"] for e in ENC]
    ax.bar(x + off, vals, w, color=col, label=lab, zorder=3,
           edgecolor=[T.PROV if p else "none" for p in prov],
           linewidth=[1.0 if p else 0 for p in prov])
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.9, f"{v:.0f}", ha="center", fontsize=5.6, color=T.INK)
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(len(ENC) - 0.4, 26.5, "chance", ha="right", fontsize=5.6, color=T.SUB)
ax.set_xticks(x); ax.set_xticklabels([e[0] for e in ENC], fontsize=5.6)
ax.set_ylabel("Konkle 4AFC (%)"); ax.set_ylim(0, 82)
ax.legend(loc="upper left", fontsize=6)
T.clean(ax)

# ---- B: ladder across encoders -------------------------------------------------
RUNGS = ["whole-\nframe", "+ region\nMIL", "+ align\nfilter", "+ referent\nlabel"]
LAD = [("DINOv3-L BabyView", [41.0, 41.7, 45.7, 51.1], T.RED),
       ("DINOv2 off-the-shelf",
        [D.claim("ladder_pure")["value"], D.claim("ladder_region")["value"],
         D.claim("ladder_filter")["value"], D.claim("ladder_vision")["value"]], T.SUB),
       ("DINOv3-B off-the-shelf", [70.8, 72.6, 75.4, 84.6], T.GREEN)]
for name, vals, col in LAD:
    bx.plot(range(4), vals, "-o", color=col, lw=1.2, ms=3, zorder=3)
    bx.annotate(f"+{vals[-1]-vals[0]:.0f}", (0, vals[0]), (-0.35, vals[0]),
                fontsize=6, color=col, va="center", ha="right", fontweight="bold")
    bx.text(3.08, vals[-1], name.split(" ")[0], fontsize=5.8, color=col, va="center")
bx.axvspan(-0.3, 1.3, color=T.GREEN, alpha=0.05)
bx.axvspan(1.7, 3.3, color=T.BLUE, alpha=0.05)
bx.text(0.5, 39, "free", color=T.GREEN, fontsize=6, ha="center", style="italic")
bx.text(2.5, 39, "oracle", color=T.BLUE, fontsize=6, ha="center", style="italic")
bx.set_xticks(range(4)); bx.set_xticklabels(RUNGS, fontsize=5.8)
bx.set_xlim(-0.75, 4.3); bx.set_ylim(38, 90); bx.set_ylabel("Konkle 4AFC (%)")
T.clean(bx)
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
note = D.provisional_note(["ladder_pure", "ladder_region", "ladder_filter", "ladder_vision"])
if note:
    print("  NOTE fig5 (red outline / DINOv2 row):", note)
print("  NOTE fig5: panel B rungs for the two DINOv3 encoders are literals pending the re-scrape")
T.save(fig, "fig5_representation")
