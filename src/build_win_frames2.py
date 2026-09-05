"""Residual neighbor frames so the +-5 s window control can run at 1M and FULL scale.
Round 1 (build_win_frames.py) embedded the windows around the 30k/300k manifests only; at
full scale that leaves windows 70% complete (mean 9.6/11). This lists every frame within
+-5 s of ANY pair frame in the corpus that is not yet in a cache (pair frames themselves are in
the main caches; round-1 neighbors in emb_win5), minus frames that are not on disk.
Writes manifests/bv26a_win5_frames2.parquet (video_id, frame_idx).
usage: python src/build_win_frames2.py"""
import os
import pandas as pd

FR = os.environ.get("BABYVIEW_FRAMES", "/ccn2b/dataset/babyview/2026.1/extracted_frames_1fps")
P = pd.read_parquet("manifests/bv26_pairs_en_audio.parquet", columns=["video_id", "frame_idx"])
have = set(zip(P.video_id, P.frame_idx.astype(int)))
R1 = pd.read_parquet("manifests/bv26a_win5_frames.parquet")
have |= set(zip(R1.video_id, R1.frame_idx.astype(int)))
man = pd.read_parquet("manifests/bv26a_base.parquet", columns=["video_id", "frame_idx"])
need = set()
for v, f in zip(man.video_id, man.frame_idx.astype(int)):
    for o in range(-5, 6):
        if f + o >= 0:
            need.add((v, f + o))
miss = sorted(need - have)
print(f"window frames for full corpus {len(need):,} | already listed {len(need & have):,} | residual {len(miss):,}")
ok = [(v, f) for v, f in miss if os.path.exists(f"{FR}/{v}/{f:05d}.jpg")]
print(f"on disk {len(ok):,} ({100 * len(ok) / max(len(miss), 1):.1f}%)")
out = pd.DataFrame(ok, columns=["video_id", "frame_idx"])
out["frame_idx"] = out.frame_idx.astype("int64")
out.to_parquet("manifests/bv26a_win5_frames2.parquet", index=False)
print(f"wrote manifests/bv26a_win5_frames2.parquet: {len(out):,} frames, {out.video_id.nunique():,} videos")
