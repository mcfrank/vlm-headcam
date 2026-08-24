"""M5: regenerate the pose flat table keyed like every other layer.

pose_1fps_bbox_limbs.csv keys on `superseded_gcp_name_feb25` (the feb25-era GCS name) +
H:MM:SS timestamps. Output: pose_1fps_bbox_limbs.parquet with
  video_id     current release name (via immutable rec-id -> release_index crosswalk)
  frame_idx    seconds (= 1 fps frame index)
  person_idx   0..n-1 within (video_id, frame_idx), by row order
  rotated      bool, from the release name
  source_name  the original feb25 name (provenance; NOT a join key)
plus every original bbox/score column. The rec-id -> release-name map is asserted 1:1 and
rows whose rec-id is not in the release are dropped WITH A COUNT (the 3 quarantined
post-release videos + anything unexpected).

usage: python src/pose_rekey.py [--csv ...] [--index ...] [--out ...]
"""
import argparse
import re

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="/ccn2b/dataset/babyview/2026.1/outputs/pose_1fps_bbox_limbs.csv")
ap.add_argument("--index", default="/ccn2b/dataset/babyview/2026.1/outputs/release_index.tsv")
ap.add_argument("--out", default="/ccn2b/dataset/babyview/2026.1/outputs/pose_1fps_bbox_limbs.parquet")
a = ap.parse_args()

idx = pd.read_csv(a.index, sep="\t")
assert idx.rec_id.is_unique
rec2name = idx.set_index("rec_id").video_id.to_dict()
REC = re.compile(r"(rec[A-Za-z0-9]{14,})")

parts, dropped, drop_ids = [], 0, set()
for ch in pd.read_csv(a.csv, chunksize=4_000_000, low_memory=False):
    r = ch.superseded_gcp_name_feb25.astype(str).str.extract(REC)[0]
    ch["video_id"] = r.map(rec2name)
    bad = ch.video_id.isna()
    dropped += int(bad.sum()); drop_ids |= set(ch.loc[bad, "superseded_gcp_name_feb25"].unique())
    ch = ch[~bad].copy()
    t = ch.time_in_extended_iso.astype(str).str.split(":")
    ch["frame_idx"] = (t.str[0].astype(int) * 3600 + t.str[1].astype(int) * 60
                       + t.str[2].astype(float).astype(int))
    ch["person_idx"] = ch.groupby(["video_id", "frame_idx"]).cumcount()
    ch["rotated"] = ch.video_id.str.endswith("_rotated")
    ch = ch.rename(columns={"superseded_gcp_name_feb25": "source_name"})
    front = ["video_id", "frame_idx", "person_idx", "rotated", "source_name"]
    parts.append(ch[front + [c for c in ch.columns if c not in front + ["time_in_extended_iso"]]])
    print(f"  +{len(ch):,} rows ({dropped:,} dropped so far)", flush=True)

out = pd.concat(parts, ignore_index=True)
out.to_parquet(a.out, index=False)
print(f"wrote {a.out}: {len(out):,} rows, {out.video_id.nunique():,} videos")
print(f"dropped {dropped:,} rows from {len(drop_ids)} non-release videos: {sorted(drop_ids)[:5]}")
assert len(drop_ids) <= 5, "more non-release videos in the pose CSV than the 3 known — investigate"
