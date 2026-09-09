"""Copy the sampled frames into the app data dir under opaque ids and write items.json.

data/frames/<item_id>.jpg   — the 1-fps midpoint frame, as stored on disk (512x910)
data/items.json             — [{"id", "text"}] : ONLY what the rater may see. No scores,
                              no referents, no video/child ids.
Gemini's answer stays in sample.parquet, which never enters the data dir.

Usage (node):
    /data2/mcfrank/gemini_check/venv/bin/python human_check/pull_frames.py \
        --sample /data2/mcfrank/gemini_check/sample.parquet --data /data2/mcfrank/gemini_check/data
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

LOCAL = Path("/data2/mcfrank/frames_1fps_local")
CANON = Path("/ccn2b/dataset/babyview/2026.1/extracted_frames_1fps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    s = pd.read_parquet(args.sample)
    out = Path(args.data) / "frames"
    out.mkdir(parents=True, exist_ok=True)
    sizes = set()
    for r in s.itertuples():
        rel = f"{r.video_id}/{int(r.frame_idx):05d}.jpg"
        src = LOCAL / rel if (LOCAL / rel).exists() else CANON / rel
        dst = out / f"{r.item_id}.jpg"
        if not dst.exists():
            shutil.copyfile(src, dst)
        sizes.add(Image.open(dst).size)
    items = [{"id": r.item_id, "text": r.text} for r in s.itertuples()]
    (Path(args.data) / "items.json").write_text(json.dumps(items, indent=0))
    print(f"{len(items)} frames in {out}, sizes {sizes}; items.json written")


if __name__ == "__main__":
    main()
