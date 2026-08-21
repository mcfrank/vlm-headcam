"""Display item 1 — design and pipeline.

A: the corpus and how an (utterance, frame) pair is formed — dense 1 fps frames, midpoint pairing,
   no score selects the frame.
B: what the referential annotation adds, and the funnel from utterances to referential moments.
C: the frozen two-tower learner and the out-of-corpus 4AFC evaluation.

Schematic — numbers come from results/corpus.csv so the counts stay in sync with the data.
"""
import sys, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
C = pd.read_csv(R / "corpus.csv").set_index("key").value.to_dict()


def box(ax, x, y, w, h, text, fc="#f2f1ec", ec=T.SUB, fs=6, bold=False, tc=T.INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=T.SUB):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6,
                                 color=color, lw=0.7, zorder=1, shrinkA=0, shrinkB=0))


fig = plt.figure(figsize=(T.W2, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.25], wspace=0.12)
axes = [fig.add_subplot(gs[i]) for i in range(3)]
for a in axes:
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")

# ---- A: corpus -> pair ---------------------------------------------------------
a = axes[0]
box(a, .06, .86, .88, .10, f"BabyView 2025.2\n{C['videos']:,.0f} videos · {C['children']:,.0f} children", fc="#e7eef6", ec=T.BLUE)
box(a, .02, .66, .46, .12, f"dense 1 fps frames\n{C['frames_total']:,.0f}", fc="#e8f4ef", ec=T.GREEN, fs=5.6)
box(a, .52, .66, .46, .12, f"time-aligned ASR\n{C['utterances']:,.0f} utterances", fs=5.6)
arrow(a, .35, .86, .25, .78); arrow(a, .65, .86, .75, .78)
box(a, .18, .44, .64, .12, "pair at the utterance\nMIDPOINT second", fc="#fff", ec=T.INK)
arrow(a, .25, .66, .40, .56); arrow(a, .75, .66, .60, .56)
a.text(.5, .365, "no alignment score selects the frame", ha="center", fontsize=5.6,
       color=T.SUB, style="italic")
box(a, .18, .20, .64, .11, f"{C['pairs']:,.0f} pairs\n({C['train_pairs']:,.0f} train on the 80% video split)",
    fc="#f2f1ec", ec=T.SUB, bold=True)
arrow(a, .5, .44, .5, .31)
a.set_title("Corpus to pairs", fontsize=7.5, color=T.INK)

# ---- B: annotation funnel ------------------------------------------------------
b = axes[1]
box(b, .06, .86, .88, .10, "Gemini-2.5-Flash (Vertex)\nper pair: alignment 0–100 + referent noun", fc="#e7eef6", ec=T.BLUE)
tot = C["pairs"]
levels = [("all pairs", tot, "#dcdad2"),
          ("about something visible", C["aligned_pairs"], T.GREEN),
          ("…and the referent is spoken", C["aligned_spoken"], T.BLUE)]
y = .60
for lab, n, col in levels:
    wfrac = .86 * (n / tot) ** 0.42
    b.add_patch(plt.Rectangle((.5 - wfrac / 2, y), wfrac, .085, fc=col, ec="none", zorder=2))
    b.text(.5, y + .0425, f"{n:,.0f}", ha="center", va="center", fontsize=6,
           color="white" if col != "#dcdad2" else T.INK, fontweight="bold", zorder=3)
    b.text(.5, y - .035, f"{lab}  ({100*n/tot:.0f}%)", ha="center", fontsize=5.8, color=T.SUB)
    y -= .20
b.text(.5, .06, "only ~1 in 11 utterances is\nabout something the child can see",
       ha="center", fontsize=6, color=T.INK, style="italic")
b.set_title("Referential annotation", fontsize=7.5, color=T.INK)

# ---- C: model + eval -----------------------------------------------------------
c = axes[2]
box(c, .04, .80, .43, .11, "frame", fc="#e8f4ef", ec=T.GREEN)
box(c, .53, .80, .43, .11, "utterance", fc="#e7eef6", ec=T.BLUE)
box(c, .04, .60, .43, .13, "frozen DINOv2\nCLS + 4×4 grid", fc="#e8f4ef", ec=T.GREEN)
box(c, .53, .60, .43, .13, "bag-of-words\n(trained from scratch)", fc="#e7eef6", ec=T.BLUE)
arrow(c, .255, .80, .255, .73); arrow(c, .745, .80, .745, .73)
box(c, .16, .40, .68, .12, "score = max over regions\n(multiple-instance)", fc="#fdf0ec", ec=T.RED)
arrow(c, .255, .60, .38, .52); arrow(c, .745, .60, .62, .52)
box(c, .16, .24, .68, .09, "InfoNCE", fc="#fff", ec=T.INK)
arrow(c, .5, .40, .5, .33)
box(c, .06, .05, .88, .12, f"evaluation: {C['konkle_cats']:.0f}-way Konkle 4AFC\nout-of-corpus object photos · chance 25%",
    fc="#f2f1ec", ec=T.SUB, bold=True)
arrow(c, .5, .24, .5, .17)
c.set_title("Frozen two-tower learner", fontsize=7.5, color=T.INK)

for a, l in zip(axes, "ABC"):
    a.text(-0.02, 1.06, l, transform=a.transAxes, fontsize=9, fontweight="bold", color=T.INK)
T.save(fig, "fig1_pipeline")
