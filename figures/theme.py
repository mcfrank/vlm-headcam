"""Shared style for paper display items. PNAS geometry: 1-col 3.42in, 1.5-col 4.5in, 2-col 7.0in.

Every figure script imports from here so type sizes, colors and output conventions are identical
across display items. Figures are written as PDF (vector, for the manuscript) + PNG (for preview).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "out"

# PNAS column widths (inches)
W1, W15, W2 = 3.42, 4.5, 7.0

# ---------------------------------------------------------------- palette
# Okabe–Ito-based scheme with the MAIN CONTRAST on the blue vs orange axis (no red/green
# opposition anywhere) — safe under deutan/protan vision, ordered in lightness within each
# family. Colors are assigned SEMANTIC roles and must stay consistent across every display
# item: the same idea is always the same color, different kinds of thing are always
# different colors.
OI = dict(blue="#0072B2", sky="#56B4E9", vermillion="#D55E00", orange="#E69F00",
          purple="#AA4499", sand="#DDCC77", wine="#882255", grey="#BBBBBB")

# ---------------------------------------------------------------- encoders
# The six frozen encoders: three sizes x two pre-training regimes. Labels are parameter
# counts (not ViT letters) for non-ML readers. Hue = regime (blues off-the-shelf, oranges
# BabyView-trained), lightness = size (darker = larger), marker = size, shared across
# regimes so the three near-coincident BabyView curves stay tellable apart.
# `legacy` is the S/B/L key still used in results/lev_scaling_final.csv; `lex` the lexicon
# family stem in results/lexicon_*.csv.
ENCODERS = [
    dict(tag="dinov3l", label="OTS-304M", legacy="L-OTS", regime="OTS", size=304, marker="^",
         color="#004070", desc="DINOv3 ViT-L/16, off-the-shelf"),
    dict(tag="dinov3b", label="OTS-86M",  legacy="B-OTS", regime="OTS", size=86,  marker="s",
         color=OI["blue"], desc="DINOv3 ViT-B/16, off-the-shelf"),
    dict(tag="dinov3s", label="OTS-22M",  legacy="S-OTS", regime="OTS", size=22,  marker="o",
         color=OI["sky"], desc="DINOv3 ViT-S/16, off-the-shelf"),
    dict(tag="vitl_bv", label="BV-304M",  legacy="L-BV",  regime="BV",  size=304, marker="^",
         color="#9A3F00", desc="ViT-L/16, BabyView-trained"),
    dict(tag="vitb_bv", label="BV-86M",   legacy="B-BV",  regime="BV",  size=86,  marker="s",
         color=OI["vermillion"], desc="ViT-B/16, BabyView-trained"),
    dict(tag="vits_bv", label="BV-22M",   legacy="S-BV",  regime="BV",  size=22,  marker="o",
         color=OI["orange"], desc="ViT-S/16, BabyView-trained"),
]
for _e in ENCODERS:
    _e["lex"] = f"F-{_e['tag']}"
ENC_BY_TAG = {e["tag"]: e for e in ENCODERS}


def enc(tag):
    """The registry entry for one encoder tag (e.g. 'dinov3l')."""
    return ENC_BY_TAG[tag]


FREE    = enc("dinov3b")["color"]  # unaided / free learning; the workhorse (OTS-86M) encoder
OTHER   = enc("dinov3l")["color"]  # the largest off-the-shelf encoder (OTS-304M)
INDOM   = enc("vitb_bv")["color"]  # in-domain (BabyView-trained) encoders — the negative result
INDOM2  = enc("vits_bv")["color"]  # the smallest BabyView-trained encoder
ORACLE  = OI["purple"]     # oracle / referential-alignment information the learner is handed
ORACLE2 = "#7A2E68"        # a nested subset of it (referent spoken subset of aligned)
LANG    = "#009E73"        # the language stream in the schematic (Okabe-Ito bluish green)
CHILD   = OI["sand"]       # human children / external reference (lighter than the oranges)
LIT     = "#8a8a86"        # published reference points from the literature (Vong, CVCL)
NEUTRAL = OI["grey"]       # a baseline / reference when it is not the point
PROV    = OI["wine"]       # provisional: source runs deleted (notes/PROVENANCE.md D6)

# legacy aliases so existing scripts keep working
GREEN, BLUE, RED, AMBER, PURPLE = FREE, ORACLE, INDOM, CHILD, LIT
INK, SUB, GRID = "#2c2c2a", "#6b6a66", "#d9d8d1"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": SUB, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,       # editable text in the PDF
})


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    return ax


def panel(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left", color=INK)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/out/{name}.pdf + .png")


def end_labels(ax, xs, ys, labels, colors, gap, xl, fontsize=5.4, ha="left", leader=True,
               va="center"):
    """Direct labels at the right end of several curves, pushed apart vertically so that
    none overlap (rank order is preserved; each colliding cluster is re-centred on its
    points). A hairline leader joins a label to its curve end when it had to move.
    xs/ys: the curve end points; xl: the x at which the labels sit (scalar or per-label)."""
    import numpy as np
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    xl = np.broadcast_to(np.asarray(xl, float), xs.shape)
    order = np.argsort(ys); y = ys[order].copy()
    for _ in range(500):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < gap - 1e-9:
                d = (gap - (y[i] - y[i - 1])) / 2
                y[i] += d; y[i - 1] -= d; moved = True
        if not moved:
            break
    yl = np.empty_like(y); yl[order] = y
    for x0, y0, x1, y1, lab, col in zip(xs, ys, xl, yl, labels, colors):
        ax.text(x1, y1, lab, fontsize=fontsize, color=col, ha=ha, va=va, zorder=7)
        if leader and abs(y1 - y0) > 0.35 * gap:
            ax.annotate("", xy=(x0, y0), xytext=(x1, y1), zorder=6,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.45, alpha=0.8,
                                        shrinkA=1.5, shrinkB=1.5))
    return yl
