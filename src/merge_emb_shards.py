"""M7: merge the 8 strided embedding shards into one emb.f16.npy + index.parquet.

Verifies before writing: shard row counts sum to the frames manifest exactly; no duplicate
(video_id, frame_idx); spot-checks 5 vectors per shard byte-identical after the merge.
Shards are NOT deleted here (retire after Oak).

usage: python src/merge_emb_shards.py [--dir ...] [--frames ...]
"""
import argparse
import glob

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/dinov3b_grid4x4")
ap.add_argument("--frames", default="/data2/mcfrank/vlm-headcam/manifests/bv2026_frames.parquet")
a = ap.parse_args()

want = pd.read_parquet(a.frames)[["video_id", "frame_idx"]].drop_duplicates()
shards = sorted(glob.glob(f"{a.dir}/shard_*"))
assert len(shards) == 8, f"expected 8 shards, found {len(shards)}"
idxs = [pd.read_parquet(f"{s}/index.parquet") for s in shards]
n = sum(len(i) for i in idxs)
print(f"shards: {[len(i) for i in idxs]} -> {n:,} (manifest {len(want):,})")
assert n == len(want), "shard union != frames manifest — a shard is incomplete"

embs = [np.load(f"{s}/emb.f16.npy", mmap_mode="r") for s in shards]
D = embs[0].shape[1:]
out = np.lib.format.open_memmap(f"{a.dir}/emb.f16.npy", mode="w+", dtype=np.float16,
                                shape=(n, *D))
pos, offsets = 0, []
for e in embs:
    out[pos:pos + len(e)] = e[:]
    offsets.append(pos); pos += len(e)
out.flush()
merged = pd.concat(idxs, ignore_index=True)
merged["row"] = np.arange(n)
assert not merged.duplicated(["video_id", "frame_idx"]).any(), "duplicate frame across shards"
merged.to_parquet(f"{a.dir}/index.parquet", index=False)

chk = np.load(f"{a.dir}/emb.f16.npy", mmap_mode="r")
rng = np.random.default_rng(0)
for si, (e, off) in enumerate(zip(embs, offsets)):
    for j in rng.integers(0, len(e), 5):
        assert np.array_equal(chk[off + j], e[j]), f"merge corruption shard {si} row {j}"
print(f"merged + verified: {a.dir}/emb.f16.npy {chk.shape}, index.parquet {len(merged):,} rows")
