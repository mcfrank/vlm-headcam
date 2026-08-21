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

# project palette — kept identical to the book so talk/book/paper read as one project
GREEN = "#1d9e75"   # free / unaided learning
BLUE = "#185fa5"    # oracle / supervised
RED = "#b0655a"     # negative results, BabyView-trained encoders
AMBER = "#b8860b"
PURPLE = "#7b5ea7"
INK, SUB, GRID = "#2c2c2a", "#6b6a66", "#d9d8d1"
PROV = "#c25b4e"    # provisional (unrecoverable provenance) — see notes/PROVENANCE.md

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
