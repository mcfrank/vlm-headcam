"""Pre-stage training frames for JOINT encoder training: apply the DINOv3 image processor's
geometric transform (resize -> 224x224) once, store uint8 jpgs on node-local NVMe. Training
then does jpg -> tensor/255 -> normalize, identical to the processor pipeline modulo jpg
requantisation. NFS is read once, here, instead of every epoch.

usage: python src/cache_frames224.py --manifests m1.parquet,m2.parquet --out /data2/mcfrank/frames224
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor

from common import frame_path

ap = argparse.ArgumentParser()
ap.add_argument("--manifests", required=True, help="comma-separated pair manifests; frames = union")
ap.add_argument("--out", default="/data2/mcfrank/frames224")
ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
ap.add_argument("--workers", type=int, default=24)
a = ap.parse_args()

proc = AutoImageProcessor.from_pretrained(a.model)
frames = pd.concat([pd.read_parquet(m)[["video_id", "frame_idx"]] for m in a.manifests.split(",")]
                   ).drop_duplicates().reset_index(drop=True)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
print(f"{len(frames):,} unique frames -> {OUT}")

fails = []
def one(r):
    dst = OUT / r.video_id / f"{int(r.frame_idx):06d}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return
    try:
        img = Image.open(frame_path(r.video_id, r.frame_idx)).convert("RGB")
        # geometric transform only; normalization stays in the training loop
        px = proc(images=img, do_normalize=False, do_rescale=False,
                  return_tensors="pt").pixel_values[0].numpy()   # fast processors are pt-only
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.transpose(px, (1, 2, 0)).astype(np.uint8)).save(dst, quality=95)
    except Exception as e:
        fails.append((r.video_id, int(r.frame_idx), str(e)[:60]))

with ThreadPoolExecutor(a.workers) as ex:
    for i, _ in enumerate(ex.map(one, frames.itertuples(index=False))):
        if i % 20000 == 0:
            print(f"  {i:,}/{len(frames):,} ({len(fails)} failed)", flush=True)
pd.DataFrame(fails, columns=["video_id", "frame_idx", "err"]).to_csv(OUT / "cache_failures.tsv",
                                                                     sep="\t", index=False)
print(f"CACHE224_DONE: {len(frames) - len(fails):,} ok, {len(fails)} failed")
