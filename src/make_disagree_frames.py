"""Render frames where CLIP and Gemini most disagree, with both judgments burned into a
header band, for local human review. Two groups: CLIP-high/Gemini-low and Gemini-high/
CLIP-low. Aggregate review artifact — written to a local (non-repo) dir, never committed."""
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

from common import frame_path

pool = pd.read_parquet("scored/pool_flash.parquet")
pool = pool[pool.alignment.notna() & pool.clip_score_max.notna()]

# group A: CLIP thinks aligned, Gemini says no visible referent
A = (pool[pool.alignment <= 2].sort_values("clip_score_max", ascending=False)
     .drop_duplicates("text").head(30))
# group B: Gemini very confident, CLIP scores low
B = (pool[pool.alignment >= 90].sort_values("clip_score_max", ascending=True)
     .drop_duplicates("text").head(30))

fp = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
F_TXT = ImageFont.truetype(fp, 24)
F_SC = ImageFont.truetype(fp, 22)
GREEN, RED, WHITE = (90, 210, 120), (235, 110, 110), (240, 240, 240)

outdir = Path("scratch/disagree")
outdir.mkdir(parents=True, exist_ok=True)


def seg(d, x, y, parts, font):
    """draw colored segments left-to-right; parts = [(text, color), ...]"""
    for t, c in parts:
        d.text((x, y), t, fill=c, font=font)
        x += d.textlength(t, font=font) + 14


def render(row, path, clip_high):
    try:
        im = Image.open(frame_path(row.video_id, row.frame_idx)).convert("RGB")
    except Exception as e:
        print("skip", row.video_id, row.frame_idx, e); return False
    W = 680
    im = im.resize((W, round(im.height * W / im.width)))
    lines = textwrap.wrap(f'"{row.text}"', width=46)[:2]
    H = 16 + 30 * len(lines) + 40
    canvas = Image.new("RGB", (W, im.height + H), (24, 24, 26))
    canvas.paste(im, (0, H))
    d = ImageDraw.Draw(canvas)
    y = 12
    for ln in lines:
        d.text((10, y), ln, fill=WHITE, font=F_TXT); y += 30
    cc = GREEN if clip_high else RED
    gc = RED if clip_high else GREEN
    seg(d, 10, y + 6, [(f"CLIP {row.clip_score_max:.3f}", cc),
                       (f"Gemini {int(row.alignment)}", gc),
                       (f"ref: {row.referent or '-'}", WHITE)], F_SC)
    canvas.save(path, quality=88)
    return True


nA = sum(render(r, outdir / f"A_clipHi_gemLo_{i:02d}.jpg", True) for i, (_, r) in enumerate(A.iterrows()))
nB = sum(render(r, outdir / f"B_gemHi_clipLo_{i:02d}.jpg", False) for i, (_, r) in enumerate(B.iterrows()))
print(f"wrote {nA} clip-high/gemini-low + {nB} gemini-high/clip-low -> {outdir}")
