"""Neighbor frames for the +-5 s temporal-window control: union of windows around the pair
frames of rand_30000 x5 seeds and rand_300000 x3 seeds, minus frames already embedded."""
import pandas as pd
P = pd.read_parquet("manifests/bv26_pairs_en_audio.parquet", columns=["video_id", "frame_idx"])
have = set(zip(P.video_id, P.frame_idx))
mans = [f"manifests/bv26a_rand_30000_s{s}.parquet" for s in range(5)] + \
       [f"manifests/bv26a_rand_300000_s{s}.parquet" for s in range(3)]
fr = set()
for m in mans:
    d = pd.read_parquet(m, columns=["video_id", "frame_idx"])
    for v, f in zip(d.video_id, d.frame_idx):
        for k in range(f - 5, f + 6):
            if k >= 0: fr.add((v, k))
new = sorted(fr - have)
out = pd.DataFrame(new, columns=["video_id", "frame_idx"])
out.to_parquet("manifests/bv26a_win5_frames.parquet", index=False)
print(f"window union {len(fr):,} | new frames to embed {len(out):,} -> manifests/bv26a_win5_frames.parquet")
