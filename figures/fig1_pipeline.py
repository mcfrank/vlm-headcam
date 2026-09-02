"""Display item 1 — design and pipeline, drawn with the real materials.

A: the corpus and how an (utterance, frame) pair is formed. A run of 1 fps frames from one
   recording with the time-aligned utterances underneath; each utterance is paired with the frame
   at its midpoint second. No score selects the frame.
B: the frozen two-tower learner (region grid over a frozen encoder; bag-of-words over the
   utterance; max-over-regions score; InfoNCE) and the out-of-corpus 4AFC evaluation.
C: what the referential annotation adds — two pairs Gemini marks as referential, two it does
   not — and the funnel from all pairs to referential moments.

All frames are face-blurred copies (src/blur_faces.py) staged in figures/assets/frames; counts
come from results/corpus.csv.
"""
import sys, json, pandas as pd, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

HERE = __import__("pathlib").Path(__file__).resolve().parent
R = HERE.parent / "results"
A = HERE / "assets"
C = pd.read_csv(R / "corpus.csv").set_index("key").value.to_dict()
J = json.loads((R / "pipeline_counts.json").read_text())
STEP = {st["step"]: st for st in J["steps"]}

# architecture constants for panel B, from runs/F_dinov3b_base_s0/metrics.json + RegionMIL:
# frozen DINOv3-B (86M) emits 1 whole-image + 4x4 region vectors at 768-d; both towers are
# projected/embedded into a shared 512-d space; the word table is the only large learned part.
ENC_PARAM_M, EMB_D, PROJ_D, N_REG, VOCAB = 86, 768, 512, 17, 15608
TEXT_PARAM_M = VOCAB * PROJ_D / 1e6
VIS_PARAM_M = (EMB_D * PROJ_D + PROJ_D + 2 * EMB_D) / 1e6

CAT_VID = "S00400001_2023-08-08_1_recSJfNNDLfxCQail"
STRIP = list(range(603, 610))                                 # seconds shown in panel A
UTTS = [(604.229, 605.070, "Wait, is that your cat?"),        # from merged_transcripts_parsed
        (606.502, 607.123, "Which cat is it?"),
        (608.567, 609.028, "That's Poe.")]
CARDS = [("S00320003_2025-06-27_3_recn38XpC15e0nbXX", 369, "Blinker, blinker, little car.", 100, "car", "bottom"),
         ("S00380001_2025-07-08_1_reciK6XREODFdF0Zz", 695, "Come to Bobo, I mean, ball.", 100, "ball", "center"),
         ("S00240001_2025-05-12_3_recxaqvlc4HaUaW5H", 758, "Pria, try your fork.", 0, None, "center"),
         ("S00400003_2024-05-23_2_recJER5vIIPF6FSeP", 123, "Let's call him later.", 0, None, "center")]
KONKLE = [("cat", "ACAT6.jpg"), ("ball", "ball8.JPG"), ("dog", "Adog120.jpg"), ("hat", "Ahat47.jpg")]

FR = lambda vid, i: A / "frames" / f"{vid}_{i:05d}.jpg"


def load(path, crop=None, anchor="center", crop_px=None):
    im = Image.open(path).convert("RGB")
    if crop_px:
        im = im.crop(crop_px)
    if crop:                                               # crop to w:h ratio
        w, h = im.size; tw, th = crop
        if w / h > tw / th: nw, nh = int(h * tw / th), h
        else: nw, nh = w, int(w * th / tw)
        x0 = (w - nw) // 2
        y0 = {"center": (h - nh) // 2, "bottom": h - nh, "top": 0}[anchor]
        im = im.crop((x0, y0, x0 + nw, y0 + nh))
    return np.asarray(im)


def img(ax, arr, x, y, w, h=None, ec=None, lw=0.6, z=2):
    """Draw an image with its top-left at (x, y) in axis units, width w (height from aspect)."""
    H, W = arr.shape[:2]
    h = h or w * H / W
    ax.imshow(arr, extent=(x, x + w, y - h, y), zorder=z, interpolation="bilinear")
    if ec:
        ax.add_patch(Rectangle((x, y - h), w, h, fc="none", ec=ec, lw=lw, zorder=z + 1))
    return x + w, y - h


def arrow(ax, x1, y1, x2, y2, color=T.SUB, lw=0.7, z=1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=6,
                                 color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def chip(ax, x, y, w, h, text, fc, ec, fs=5.4, tc=T.INK, bold=False, z=3, pad=0.25):
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
                                fc=fc, ec=ec, lw=0.6, zorder=z))
    ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=z + 1, linespacing=1.3)


# ---------------------------------------------------------------- canvas (units = 0.1 in)
FW, FH = T.W2, 4.3
fig = plt.figure(figsize=(FW, FH))
PANELS = {"A": (0.00, 2.25), "B": (2.35, 2.40), "C": (4.85, 2.15)}     # x-offset, width in inches
axes = {}
for k, (x0, w) in PANELS.items():
    ax = fig.add_axes([x0 / FW, 0, w / FW, 1])
    ax.set_xlim(0, w * 10); ax.set_ylim(0, FH * 10); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, FH * 10 - 0.5, k, fontsize=9, fontweight="bold", color=T.INK, va="top")
    axes[k] = ax
TOP = FH * 10 - 3.2

# ================================================================ A: corpus -> pairs
ax = axes["A"]
y = TOP
cam = A / "camera.png"
if cam.exists():
    _, cam_bot = img(ax, load(cam, crop_px=(250, 150, 1408, 1700)), 1.0, y, 6.4)
else:
    print("  NOTE fig1: figures/assets/camera.png missing — camera inset skipped")
    cam_bot = y - 2.0
ax.text(8.8, y - 1.8, f"BabyView {J['release']}\n{STEP['release']['children']} children\n"
        f"{STEP['release']['videos']:,} recordings", fontsize=5.6, color=T.INK, va="top",
        linespacing=1.5)
y = cam_bot - 1.8

# frame strip: 7 consecutive seconds, 1 fps
n = len(STRIP); gap = 0.25; x0 = 1.0; usable = 22.5 - 2 * x0
fw = (usable - (n - 1) * gap) / n
fh = fw * 910 / 512
xs = {}
for i, sec in enumerate(STRIP):
    x = x0 + i * (fw + gap)
    img(ax, load(FR(CAT_VID, sec)), x, y, fw, ec="#ffffff", lw=0)
    xs[sec] = x + fw / 2
ax.text(x0, y + 0.5, "frames, 1 per second", fontsize=5.4, color=T.FREE, va="bottom")
y_strip_bot = y - fh
# time axis under the strip
ty = y_strip_bot - 1.3
sec0 = STRIP[0]; per_s = fw + gap
tx = lambda t: x0 + (t - sec0) * per_s + fw / 2 - per_s / 2 + gap / 2   # t in seconds -> x
ax.plot([x0, x0 + usable], [ty, ty], color=T.SUB, lw=0.5)
for sec in STRIP + [STRIP[-1] + 1]:
    ax.plot([tx(sec)] * 2, [ty, ty + 0.35], color=T.SUB, lw=0.5)
ax.text(x0, ty - 0.5, f"{sec0} s", fontsize=4.8, color=T.SUB, va="top")
ax.text(x0 + usable, ty - 0.5, f"{STRIP[-1] + 1} s", fontsize=4.8, color=T.SUB, va="top", ha="right")
# utterances as bars at their real times; each pairs with the frame at its midpoint second
uy = ty - 3.0
ax.text(x0, uy + 0.9, "transcribed speech", fontsize=5.4, color=T.LANG, va="bottom")
for k, (t0, t1, text) in enumerate(UTTS):
    main = k == 1
    yy = uy - (k % 2) * 1.9
    col = T.LANG if main else "#7fcdb4"
    ax.add_patch(Rectangle((tx(t0), yy - 0.55), tx(t1) - tx(t0), 1.1, fc=col, ec="none", zorder=3))
    ax.text(tx(t0) if k < 2 else tx(t1), yy - 0.95, f"“{text}”", fontsize=4.9, color=col,
            va="top", ha="left" if k < 2 else "right", zorder=3)
    mid = (t0 + t1) / 2; sec = int(mid)
    ax.plot([tx(mid), tx(mid)], [yy + 0.55, y_strip_bot - 0.2], color=col, lw=0.7,
            ls="-" if main else (0, (1.5, 1.5)), zorder=2)
    ax.plot([tx(mid), xs[sec]], [y_strip_bot - 0.2, y_strip_bot - 0.2], color=col, lw=0.7,
            ls="-" if main else (0, (1.5, 1.5)), zorder=2)
    if main:
        ax.add_patch(Rectangle((xs[sec] - fw / 2, y_strip_bot), fw, fh, fc="none", ec=T.LANG,
                               lw=1.1, zorder=4))
ax.text(x0 + usable / 2, uy - 4.6, "pair each utterance with the frame at its midpoint",
        fontsize=5.4, color=T.INK, ha="center", va="top", style="italic")
# counts, from the committed pipeline funnel (results/pipeline_counts.json)
cy = uy - 8.6
ax.text(x0 + usable / 2, cy,
        f"{STEP['transcribed']['utterances']:,} utterances  ·  "
        f"{STEP['pairs']['frames']:,} frames",
        fontsize=5.4, color=T.SUB, ha="center", va="top")
arrow(ax, x0 + usable / 2, cy - 1.6, x0 + usable / 2, cy - 3.0)
chip(ax, x0 + 2, cy - 3.2, usable - 4, 2.6,
     f"{STEP['training-corpus']['pairs']:,} training pairs", fc="#f2f1ec", ec=T.SUB,
     fs=6.2, bold=True)
ef = STEP["english-filter"]
ax.text(x0 + usable / 2, cy - 6.3,
        f"after English filter: −{ef['videos_dropped']:,} videos, {ef['children']} children",
        fontsize=5.0, color=T.SUB, ha="center", va="top", style="italic")

# ================================================================ C: referential annotation
bx = axes["C"]
y = TOP
bx.text(0.8, y - 0.2, "Gemini 2.5 Flash rates each pair:\nalignment 0–100  +  referent noun",
        fontsize=5.4, color=T.INK, va="top", linespacing=1.4)
cw, ch = 9.6, 9.6
for k, (vid, fi, text, score, ref, anchor) in enumerate(CARDS):
    col, row = k % 2, k // 2
    x = 0.8 + col * (cw + 0.9)
    yy = y - 3.6 - row * (ch + 3.2)
    aligned = score >= 50
    img(bx, load(FR(vid, fi), crop=(1, 1), anchor=anchor), x, yy, cw,
        ec=T.ORACLE if aligned else T.SUB, lw=1.0)
    bx.text(x, yy - ch - 0.5, f"“{text}”", fontsize=4.9, color=T.INK, va="top")
    tag = f"aligned {score:.0f}  ·  referent: {ref}" if aligned else f"aligned {score:.0f}  ·  no referent"
    bx.text(x, yy - ch - 1.7, tag, fontsize=4.9, color=T.ORACLE if aligned else T.SUB, va="top")
# funnel: real bar chart, left-aligned, true linear scale
tot = STEP["training-corpus"]["pairs"]
levels = [("all pairs", tot, "#dcdad2", T.INK),
          ("about something visible", STEP["aligned"]["pairs"], T.ORACLE, T.ORACLE),
          ("…and the referent is spoken", STEP["referent-spoken"]["pairs"], T.ORACLE2, T.ORACLE2)]
fy = y - 2 * (ch + 3.2) - 4.8
BX0, BW, BH = 0.8, 19.8, 1.7
bx.plot([BX0, BX0], [fy - 3 * 3.1 + 1.0, fy], color=T.SUB, lw=0.6, zorder=3)   # the axis
for lab, nn, col, tc in levels:
    w = BW * nn / tot
    bx.add_patch(Rectangle((BX0, fy - BH), w, BH, fc=col, ec="none", zorder=2))
    if w > 8:
        bx.text(BX0 + w / 2, fy - BH / 2, f"{nn:,.0f}   {lab}  (100%)", ha="center",
                va="center", fontsize=5.2, color=T.INK, zorder=3)
    else:
        bx.text(BX0 + w + 0.5, fy - BH / 2, f"{nn:,.0f}   {lab}  ({100 * nn / tot:.0f}%)",
                ha="left", va="center", fontsize=5.2, color=tc, zorder=3)
    fy -= 3.1

# ================================================================ B: learner + eval
cx = axes["B"]
y = TOP
iy = y - 1.8                                       # tops of the two towers
# left tower: the frame through the frozen encoder's region grid
fwC = 6.4; fhC = fwC * 910 / 512
arr = load(FR(CAT_VID, 606))
img(cx, arr, 1.0, iy, fwC)
for i in range(1, 4):                                            # 4x4 region grid
    cx.plot([1.0 + fwC * i / 4] * 2, [iy - fhC, iy], color="white", lw=0.5, alpha=0.9, zorder=3)
    cx.plot([1.0, 1.0 + fwC], [iy - fhC * i / 4] * 2, color="white", lw=0.5, alpha=0.9, zorder=3)
cx.text(1.0, y - 0.2, "frame", fontsize=5.4, color=T.FREE, va="top")
cx.text(1.0, iy - fhC - 0.4, f"frozen encoder · {ENC_PARAM_M}M, not trained\n"
        f"whole image + 4×4 regions\n{N_REG} × {EMB_D}-d, projected to {PROJ_D}-d",
        fontsize=5.0, color=T.FREE, va="top", linespacing=1.3)
FX = 1.0 + fwC / 2                                  # frame tower centerline
# right tower: the utterance as a bag of words
ux = 10.0
cx.text(ux, y - 0.2, "utterance", fontsize=5.4, color=T.LANG, va="top")
cx.text(ux, iy, "“Which cat is it?”", fontsize=5.2, color=T.INK, va="top")
words = ["which", "cat", "is", "it"]
wy = iy - 2.2
for i, w in enumerate(words):
    chip(cx, ux + (i % 2) * 5.2, wy - (i // 2) * 2.3, 4.6, 1.8, w, fc="#e4f4ee", ec=T.LANG, fs=5.2)
cx.text(ux, wy - 5.0, f"bag of words, learned from scratch\n{VOCAB:,} words × {PROJ_D}-d · "
        f"{TEXT_PARAM_M:.1f}M params", fontsize=5.0, color=T.LANG, va="top", linespacing=1.3)
WX = ux + 4.9                                       # word tower centerline
# the Y: both routes converge on the word-region similarity map
sim = np.array([[.1, .2, .1, .1], [.2, .3, .2, .1], [.1, .9, .4, .1], [.1, .3, .2, .1]])
cell = 1.25
sx = (FX + WX) / 2 - 2 * cell
sy = 21.2
for i in range(4):
    for j in range(4):
        cx.add_patch(Rectangle((sx + j * cell, sy - (i + 1) * cell), cell, cell,
                               fc=plt.cm.Greys(0.10 + 0.72 * sim[i, j]), ec="white", lw=0.4,
                               zorder=2))
im_ = np.unravel_index(sim.argmax(), sim.shape)
cx.add_patch(Rectangle((sx + im_[1] * cell, sy - (im_[0] + 1) * cell), cell, cell, fc="none",
                       ec=T.INK, lw=1.0, zorder=3))
arrow(cx, FX, 23.6, sx + 0.7, sy + 0.15, color=T.FREE)
arrow(cx, WX, wy - 8.0, sx + 4 * cell - 0.7, sy + 0.15, color=T.LANG)
cx.text(sx + 4 * cell + 0.7, sy - 2 * cell, "word · region\nsimilarity\nscore = max\nover regions",
        fontsize=5.0, color=T.INK, va="center", ha="left", linespacing=1.35)
# InfoNCE: batch similarity matrix, diagonal = true pairs
nb = 5; nc = 1.0
nx = sx + 2 * cell - nb * nc / 2
ny = sy - 4 * cell - 2.2
rng = np.random.default_rng(1)
M = rng.uniform(0.05, 0.4, (nb, nb)); np.fill_diagonal(M, rng.uniform(0.75, 0.95, nb))
for i in range(nb):
    for j in range(nb):
        cx.add_patch(Rectangle((nx + j * nc, ny - (i + 1) * nc), nc, nc,
                               fc=plt.cm.Greys(0.1 + 0.8 * M[i, j]), ec="white", lw=0.3, zorder=2))
arrow(cx, sx + 2 * cell, sy - 4 * cell - 0.4, sx + 2 * cell, ny + 0.2)
cx.text(nx - 0.6, ny - nb * nc / 2, "frames", fontsize=4.8, color=T.FREE, ha="right",
        va="center", rotation=90)
cx.text(nx + nb * nc / 2, ny - nb * nc - 0.3, "utterances", fontsize=4.8, color=T.LANG,
        ha="center", va="top")
cx.text(nx + nb * nc + 0.7, ny - nb * nc / 2, "InfoNCE\ntrue pairs on\nthe diagonal",
        fontsize=5.0, color=T.INK, va="center", linespacing=1.3)
# evaluation: 4AFC over out-of-corpus object photos
ey = ny - nb * nc - 3.0
cx.text(1.0, ey + 0.3, f"evaluation: “cat”?   {C['konkle_cats']:.0f}-way 4AFC, out-of-corpus photos",
        fontsize=5.4, color=T.INK, va="bottom")
kw = 4.6
for i, (lab, fn) in enumerate(KONKLE):
    x = 1.0 + i * (kw + 0.9)
    img(cx, load(A / "konkle" / fn), x, ey - 0.3, kw, ec=T.FREE if lab == "cat" else "#d0cfc8",
        lw=1.2 if lab == "cat" else 0.5)
    cx.text(x + kw / 2, ey - 0.3 - kw - 0.4, lab, fontsize=5.0, color=T.INK, ha="center", va="top")
cx.text(1.0 + 4 * kw + 3 * 0.9, ey - 0.3 - kw - 1.8, "chance 25%", fontsize=5.0, color=T.SUB,
        ha="right", va="top")

T.save(fig, "fig1_pipeline")
