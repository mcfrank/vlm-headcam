"""Generate all book figures on ccn2. Result plots (aggregate, safe to commit) go to
book_figs/; frame-based explainer figures (child imagery, gitignored) go to book_figs/frames/.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image

from common import CLIP_RESULTS, DETS, MANIFEST_DIR, frame_path

W = Path("/data2/mcfrank/vlm-headcam")
RUNS = W / "runs"
OUT = W / "book_figs"; OUT.mkdir(exist_ok=True)
FOUT = OUT / "frames"; FOUT.mkdir(exist_ok=True)
C = {"aligned": "#2a7ab9", "random": "#b0b0b0", "unfiltered": "#e08a3c", "crop": "#4a9d5b"}
plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def best(name):
    p = RUNS / name / "log.json"
    if not p.exists():
        return None
    return max(r["acc_4afc"] for r in json.loads(p.read_text()))


def seeds_best(prefix):
    vals = [best(f"{prefix}_s{s}") for s in range(3)]
    vals = [v for v in vals if v]
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, 0)


# ---------- B: robustness ----------
children = ["S00360001", "S00240001", "S00370002"]
al = [seeds_best(f"B_aligned_{c}") for c in children]
rn = [seeds_best(f"B_random_{c}") for c in children]
x = np.arange(len(children)); wbar = 0.38
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.bar(x - wbar/2, [m*100 for m, _ in al], wbar, yerr=[s*100 for _, s in al],
       label="aligned (clip>0.24)", color=C["aligned"], capsize=3)
ax.bar(x + wbar/2, [m*100 for m, _ in rn], wbar, yerr=[s*100 for _, s in rn],
       label="random (size-matched)", color=C["random"], capsize=3)
ax.axhline(25, ls="--", c="k", lw=1); ax.text(len(children)-0.5, 26, "chance", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(children, rotation=15)
ax.set_ylabel("4AFC accuracy (%)"); ax.set_title("Exp B: alignment gap across held-out children (±sd, 3 seeds)")
ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig(OUT/"fig_results_B.png"); plt.close(fig)

# ---------- A: dilution ----------
a = json.loads((RUNS/"A_unfiltered_S00360001"/"log.json").read_text())
au = max(r["acc_4afc"] for r in a)
bars = [("aligned\n137k", best("B_aligned_S00360001_s0"), C["aligned"]),
        ("unfiltered\n1.14M", au, C["unfiltered"]),
        ("random\n137k", best("B_random_S00360001_s0"), C["random"])]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.bar([b[0] for b in bars], [b[1]*100 for b in bars], color=[b[2] for b in bars])
ax.axhline(25, ls="--", c="k", lw=1); ax.text(2.1, 26, "chance", fontsize=9)
for i, b in enumerate(bars):
    ax.text(i, b[1]*100+0.6, f"{b[1]*100:.1f}", ha="center", fontsize=10)
ax.set_ylabel("4AFC accuracy (%)"); ax.set_title("Exp A: 8× more unfiltered data < filtered subset")
fig.tight_layout(); fig.savefig(OUT/"fig_results_A.png"); plt.close(fig)

# ---------- C: threshold sweep ----------
pts = [(0.24, 137179, seeds_best("B_aligned_S00360001")),
       (0.26, 21877, seeds_best("C_thr0.26")),
       (0.28, 3799, seeds_best("C_thr0.28"))]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
xs = [p[1] for p in pts]; ys = [p[2][0]*100 for p in pts]; es = [p[2][1]*100 for p in pts]
ax.errorbar(xs, ys, yerr=es, marker="o", ms=8, lw=2, color=C["aligned"], capsize=3)
for thr, n, (m, _) in pts:
    ax.annotate(f"clip>{thr}\n{n:,} pairs", (n, m*100), textcoords="offset points",
                xytext=(6, 8), fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("training pairs (log)"); ax.set_ylabel("4AFC accuracy (%)")
ax.set_title("Exp C: count-vs-quality sweet spot"); fig.tight_layout()
fig.savefig(OUT/"fig_results_C.png"); plt.close(fig)

# ---------- D: 2x2 ----------
M = np.array([[best("D_frame_frameeval"), best("D_frame_cropeval")],
              [best("D_crop_frameeval"), best("D_crop_cropeval")]]) * 100
fig, ax = plt.subplots(figsize=(4.8, 4.0))
im = ax.imshow(M, cmap="Greens", vmin=35, vmax=50)
ax.set_xticks([0, 1]); ax.set_xticklabels(["frame-eval", "crop-eval"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["frame-train", "crop-train"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=14,
                color="white" if M[i, j] > 44 else "black")
ax.set_title("Exp D: object crops vs whole scenes"); fig.tight_layout()
fig.savefig(OUT/"fig_results_D.png"); plt.close(fig)
print("result plots done")


# ---------- frame explainer: high vs low alignment ----------
def show_frame(ax, vid, fidx, title, border=None):
    try:
        img = Image.open(frame_path(vid, fidx)).convert("RGB")
        ax.imshow(img)
    except Exception:
        ax.text(0.5, 0.5, "frame missing", ha="center")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if border:
        for s in ax.spines.values():
            s.set_color(border); s.set_linewidth(3)


cr = pd.read_csv(CLIP_RESULTS, usecols=["video_name", "utterance",
     "utterance_start_time", "utterance_end_time", "clip_score_max"]).dropna()
cr["frame_idx"] = ((cr.utterance_start_time + cr.utterance_end_time)/2).astype(int)
cr["nw"] = cr.utterance.str.split().str.len()
cand = cr[(cr.nw >= 2) & (cr.nw <= 7)]
hi = cand.nlargest(400, "clip_score_max").sample(4, random_state=1)
lo = cand[cand.clip_score_max < 0.19].sample(4, random_state=1)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, r in zip(axes[0], hi.itertuples()):
    show_frame(ax, r.video_name, r.frame_idx,
               f'"{r.utterance.strip()}"\nclip={r.clip_score_max:.2f}', border="#2a7ab9")
for ax, r in zip(axes[1], lo.itertuples()):
    show_frame(ax, r.video_name, r.frame_idx,
               f'"{r.utterance.strip()}"\nclip={r.clip_score_max:.2f}', border="#c0392b")
fig.suptitle("Top: high frame–utterance alignment (kept by filter)   |   "
             "Bottom: low alignment (dropped)", fontsize=12)
fig.tight_layout(); fig.savefig(FOUT/"fig_alignment_examples.png"); plt.close(fig)

# ---------- 4AFC trial example ----------
ev = pd.read_parquet(MANIFEST_DIR/"eval_frames.parquet")
rng = np.random.default_rng(3)
cat = "car" if (ev.category == "car").any() else ev.category.value_counts().index[0]
tgt = ev[ev.category == cat].sample(1, random_state=3).iloc[0]
others = ev[ev.category != cat].groupby("category").sample(1, random_state=3)
dist = others.sample(3, random_state=3)
trial = [(tgt.video_id, tgt.frame_idx, tgt.category, True)] + \
        [(d.video_id, d.frame_idx, d.category, False) for d in dist.itertuples()]
order = rng.permutation(4)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, k in zip(axes, order):
    v, f, c, is_t = trial[k]
    show_frame(ax, v, f, ("TARGET" if is_t else "distractor"),
               border="#2a7ab9" if is_t else None)
fig.suptitle(f'4AFC trial — cue word: "{cat}"  (model picks the frame whose embedding is '
             f'closest to the word embedding; chance = 25%)', fontsize=11)
fig.tight_layout(); fig.savefig(FOUT/"fig_4afc_trial.png"); plt.close(fig)

# ---------- crop example ----------
rc = pd.read_parquet(MANIFEST_DIR/"region_crop_S00360001_keyed.parquet") \
    if (MANIFEST_DIR/"region_crop_S00360001_keyed.parquet").exists() else None
if rc is not None and len(rc):
    r = rc.sample(1, random_state=5).iloc[0]
    img = Image.open(frame_path(r.video_id, r.frame_idx)).convert("RGB")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2))
    a1.imshow(img); a1.add_patch(Rectangle((r.xmin, r.ymin), r.xmax-r.xmin, r.ymax-r.ymin,
              fill=False, edgecolor="#4a9d5b", lw=3))
    a1.set_title(f'scene  ·  "{r.text.strip()[:40]}"', fontsize=10); a1.set_xticks([]); a1.set_yticks([])
    crop = img.crop((int(r.xmin), int(r.ymin), int(r.xmax), int(r.ymax)))
    a2.imshow(crop); a2.set_title("object crop (what region grounding pairs)", fontsize=10)
    a2.set_xticks([]); a2.set_yticks([])
    fig.tight_layout(); fig.savefig(FOUT/"fig_crop_example.png"); plt.close(fig)
print("frame figures done ->", FOUT)

# ---------- topline: alignment injection on fixed frames ----------
def sb(prefix):
    vals = [best(f"{prefix}_s{s}") for s in range(3)]
    vals = [v for v in vals if v]
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, 0)


tl = [("random\n(Exp B)", sb("B_random_S00360001"), C["random"]),
      ("control\n(utt only)", sb("T_control"), "#8fb0c9"),
      ("+label\n(injected)", sb("T_label"), C["aligned"]),
      ("label-only\n(ceiling)", sb("T_labelonly"), C["crop"])]
if all(not np.isnan(t[1][0]) for t in tl):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar([t[0] for t in tl], [t[1][0]*100 for t in tl],
           yerr=[t[1][1]*100 for t in tl], color=[t[2] for t in tl], capsize=3)
    ax.axhline(25, ls="--", c="k", lw=1); ax.text(3.1, 26, "chance", fontsize=9)
    ax.axhline(46.9, ls=":", c=C["aligned"], lw=1.5)
    ax.text(-0.4, 47.6, "aligned (Exp B)", fontsize=8, color=C["aligned"])
    for i, t in enumerate(tl):
        ax.text(i, t[1][0]*100+0.7, f"{t[1][0]*100:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("4AFC accuracy (%)")
    ax.set_title("Topline: injecting alignment onto fixed random frames")
    fig.tight_layout(); fig.savefig(OUT/"fig_results_topline.png"); plt.close(fig)
    print("topline figure done")
