"""Assemble one of Khai's per-frame embedding readouts (one .npy per image_id) into our stacked
cache format: emb.f16.npy [N, R, D] + index.parquet (video_id, frame_idx, row), in the order of an
image-ids file. R = 1 for mean-pooled vectors (=> RegionMIL degenerates to whole-frame TwoTower),
R = 16 for _grid4x4 readouts (=> region-MIL). image_id = {video_id}_{frame_idx:05d}. Threaded reads
to survive NFS small-file latency."""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)      # dir containing per-readout subdirs
ap.add_argument("--readout", required=True)   # subdir name
ap.add_argument("--ids", required=True)       # file of image_ids (one per line), defines row order
ap.add_argument("--out", required=True)
a = ap.parse_args()


ids = [l.strip() for l in open(a.ids) if l.strip()]
if not ids:
    raise SystemExit(f"ABORT: {a.ids} is empty — nothing to assemble "
                     f"(this silently left empty cache dirs on 2026-08-22)")
rdir = f"{a.base}/{a.readout}"
os.makedirs(a.out, exist_ok=True)
first = np.load(f"{rdir}/{ids[0]}.npy")
R, D = (1, first.shape[0]) if first.ndim == 1 else first.shape   # [D]->1 region ; [16,D]->16
print(f"{a.readout}: {len(ids)} frames x {R} regions x {D}d", flush=True)

emb = np.zeros((len(ids), R, D), dtype=np.float16)


def load(i):
    emb[i] = np.load(f"{rdir}/{ids[i]}.npy").reshape(R, D)


with ThreadPoolExecutor(max_workers=32) as ex:
    for n, _ in enumerate(ex.map(load, range(len(ids)))):
        if n % 100000 == 0 and n:
            print(f"  {n}/{len(ids)}", flush=True)

vid_fi = [i.rsplit("_", 1) for i in ids]
idx = pd.DataFrame({"video_id": [v for v, _ in vid_fi],
                    "frame_idx": [int(f) for _, f in vid_fi],
                    "row": np.arange(len(ids))})
np.save(f"{a.out}/emb.f16.npy", emb)
idx.to_parquet(f"{a.out}/index.parquet", index=False)
print(f"wrote {a.out}  emb {emb.shape} ({emb.nbytes/1e9:.2f} GB)", flush=True)
