"""+-5 s window neighbor frames for the L-BV encoder (vitl_bv), split for a staggered embed:
the union of round-1 (bv26a_win5_frames) and round-2 (bv26a_win5_frames2) neighbor lists —
4.92M frames, complete windows for every manifest up to the full corpus — shuffled (seed 0) and
cut into an EARLY part (2 shards on the free GPUs now) and a LATE part (6 shards once the main
L-BV frame cache finishes). Sizes chosen so both parts finish at about the same time at the
observed ~32k frames/h/shard (early starts ~9 h before late).
Writes manifests/bv26a_win5_vitl_early.parquet and manifests/bv26a_win5_vitl_late.parquet.
usage: python src/build_win_frames_vitl.py [early_n=1650000]"""
import sys
import pandas as pd

early_n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_650_000
w = pd.concat([pd.read_parquet("manifests/bv26a_win5_frames.parquet"),
               pd.read_parquet("manifests/bv26a_win5_frames2.parquet")], ignore_index=True)
assert not w.duplicated().any(), "round-1 and round-2 neighbor lists overlap"
w = w.sample(frac=1.0, random_state=0).reset_index(drop=True)
w["frame_idx"] = w.frame_idx.astype("int64")
w.iloc[:early_n].to_parquet("manifests/bv26a_win5_vitl_early.parquet", index=False)
w.iloc[early_n:].to_parquet("manifests/bv26a_win5_vitl_late.parquet", index=False)
print(f"neighbor frames {len(w):,} -> early {early_n:,} (2 shards of {early_n // 2:,}) | "
      f"late {len(w) - early_n:,} (6 shards of {(len(w) - early_n) // 6:,})")
