"""Representative Gemini referent annotations: candidate cards (positive = high alignment with a
spoken concrete referent; negative = zero alignment, no referent) plus a 2x2 composite of the
first two of each, for local human review / talk use. Frames are human-subjects data: output goes
to scratch/ (non-repo) on ccn2 and, if pulled locally, to the gitignored book/figures/frames/.

usage: python src/make_gemini_examples.py [--pos i,j] [--neg k,l]   # indices into the candidate
lists to compose (default 0,1 / 0,1); rerun after reviewing the cards to pick better ones."""
import argparse
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

from common import frame_path


def is_portrait(row):
    try:
        w, h = Image.open(frame_path(row.video_id, row.frame_idx)).size
        return h > w
    except Exception:
        return False

ap = argparse.ArgumentParser()
ap.add_argument("--pos", default="0,1")
ap.add_argument("--neg", default="0,1")
ap.add_argument("--n", type=int, default=10, help="candidates per group")
ap.add_argument("--dry", action="store_true", help="gray boxes instead of frames (layout check)")
ap.add_argument("--clip-min", type=float, default=0.26, help="positives must also have CLIP >= this (triple agreement)")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

d = pd.read_parquet("scored/gemini_full.parquet")
d = d[d.alignment.notna()].copy()
d["nw"] = d.text.str.split().str.len()

# positive: very confident, referent word actually spoken, moderate-length utterance,
# CLIP agrees (guards against depictions / hallucinated referents), and the referent is a
# concrete physical object a child handles (not books/pictures/screens)
CONCRETE = {"ball", "cup", "spoon", "fork", "bowl", "plate", "bottle", "shoe", "sock", "hat", "banana",
            "apple", "orange", "cracker", "cookie", "block", "blocks", "car", "truck", "train", "duck",
            "bear", "doll", "baby", "dog", "cat", "bubbles", "balloon", "bucket", "brush", "towel",
            "diaper", "bib", "chair", "table", "door", "box", "bag", "key", "keys", "phone", "remote",
            "cup", "milk", "water", "juice", "egg", "cheese", "bread", "toast", "pumpkin", "flower",
            "leaf", "stick", "rock", "soap", "toothbrush", "comb", "puzzle", "lego", "legos", "bike",
            "stroller", "swing", "slide", "bath", "tub", "shirt", "pants", "jacket", "boots", "mitten"}
pos = d[(d.alignment >= 90) & (d.referent.str.len() > 0) & d.nw.between(4, 10) & (d.clip_score_max >= a.clip_min)]
pos = pos[[r.lower() in t.lower() and r.lower() in CONCRETE for r, t in zip(pos.referent, pos.text)]]
pos = pos.sample(frac=1, random_state=a.seed).drop_duplicates("referent").drop_duplicates("child_id").head(4 * a.n)
pos = pos[[is_portrait(r) for _, r in pos.iterrows()]].head(a.n)     # BabyView is portrait; keep frames uniform
# negative: no alignment, no referent, a real utterance (not a fragment)
neg = d[(d.alignment == 0) & (d.referent.str.len() == 0) & d.nw.between(4, 10)]
neg = neg.sample(frac=1, random_state=a.seed).drop_duplicates("text").drop_duplicates("child_id").head(4 * a.n)
neg = neg[[is_portrait(r) for _, r in neg.iterrows()]].head(a.n)

fp = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
F_TXT = ImageFont.truetype(fp, 26)
F_SC = ImageFont.truetype(fp, 24)
F_HDR = ImageFont.truetype(fp, 30)
INK, SUB, GREEN, RED, BG = (36, 35, 31), (107, 106, 102), (29, 158, 117), (176, 101, 90), (255, 255, 255)
W = 640

out = Path("scratch/gemini_examples"); out.mkdir(parents=True, exist_ok=True)


def card(row, positive):
    """frame + caption band below: utterance, then Gemini score + referent."""
    if a.dry:
        im = Image.new("RGB", (W, 360), (200, 200, 200))
    else:
        im = Image.open(frame_path(row.video_id, row.frame_idx)).convert("RGB")
        im = im.resize((W, round(im.height * W / im.width)))
    lines = textwrap.wrap(f"“{row.text}”", width=44)[:3]
    H = 18 + 34 * len(lines) + 44
    c = Image.new("RGB", (W, im.height + H), BG)
    c.paste(im, (0, 0))
    dr = ImageDraw.Draw(c)
    y = im.height + 12
    for ln in lines:
        dr.text((12, y), ln, fill=INK, font=F_TXT); y += 34
    col = GREEN if positive else RED
    ref = row.referent if positive else "none"
    dr.text((12, y + 4), f"Gemini alignment {int(row.alignment)}", fill=col, font=F_SC)
    x = 12 + dr.textlength(f"Gemini alignment {int(row.alignment)}", font=F_SC) + 28
    dr.text((x, y + 4), f"referent: {ref}", fill=SUB, font=F_SC)
    return c


# individual candidate cards for review
P = [card(r, True) for _, r in pos.iterrows()]
N = [card(r, False) for _, r in neg.iterrows()]
for i, c in enumerate(P): c.save(out / f"pos_{i:02d}.jpg", quality=90)
for i, c in enumerate(N): c.save(out / f"neg_{i:02d}.jpg", quality=90)
pos.assign(k=range(len(pos)))[["k", "video_id", "frame_idx", "alignment", "clip_score_max", "referent", "text"]].to_csv(out / "pos.csv", index=False)
neg.assign(k=range(len(neg)))[["k", "video_id", "frame_idx", "alignment", "referent", "text"]].to_csv(out / "neg.csv", index=False)

# 1x4 composite: two positive cards, then two negative, group headers above each pair
pi = [int(x) for x in a.pos.split(",")]; ni = [int(x) for x in a.neg.split(",")]
cards = [P[i] for i in pi] + [N[i] for i in ni]
GAP, HDR = 28, 60
cw = W; ch = max(c.height for c in cards)
comp = Image.new("RGB", (4 * cw + 5 * GAP, HDR + ch + 2 * GAP), BG)
dr = ImageDraw.Draw(comp)
for k, c in enumerate(cards):
    comp.paste(c, (GAP + k * (cw + GAP), HDR + GAP))
dr.text((GAP, 16), "Aligned: a visible referent is named", fill=GREEN, font=F_HDR)
dr.text((GAP + 2 * (cw + GAP), 16), "Not aligned: nothing visible is named", fill=RED, font=F_HDR)
comp.save(out / "fig_gemini_examples.png")
print(f"wrote {len(P)} pos + {len(N)} neg cards and composite -> {out}")
print(pos[["alignment", "clip_score_max", "referent", "text"]].round(2).to_string())
print(neg[["alignment", "text"]].to_string())
