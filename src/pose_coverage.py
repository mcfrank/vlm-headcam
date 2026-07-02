"""How much of our 2025.2 training manifest has EXISTING pose coverage on disk?
Join: manifest.video_id -> rec-id -> videos.csv[unique_video_id] -> superseded_gcp_name_feb25
-> pose dir /ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old/{name}_processed."""
import re, os
import numpy as np, pandas as pd

POSE = "/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old"
man = pd.read_parquet("/data2/mcfrank/vlm-headcam/manifests/boot_across_140000.parquet")
xw = pd.read_csv("/data2/mcfrank/vlm-headcam/metadata/videos.csv", dtype=str, low_memory=False)
xw.columns = [c.lstrip("﻿") for c in xw.columns]

def recid(v):
    m = re.search(r"rec[A-Za-z0-9]+", str(v))
    return m.group(0) if m else None

man["rec"] = man.video_id.map(recid)
xw_map = xw.set_index("unique_video_id")
gcp = xw_map["superseded_gcp_name_feb25"]
rel = xw_map["release"]; ds = xw_map["dataset"]; imu = xw_map["imu_processed"]

# video-level table
vids = man.drop_duplicates("video_id")[["video_id", "rec", "child_id"]].copy()
vids["gcp"] = vids.rec.map(gcp)
vids["release"] = vids.rec.map(rel)
vids["dataset"] = vids.rec.map(ds)
vids["imu"] = vids.rec.map(imu)
vids["in_xwalk"] = vids.rec.isin(xw_map.index)
# pose dir on disk?
posedirs = set(os.listdir(POSE)) if os.path.isdir(POSE) else set()
vids["pose_on_disk"] = vids.gcp.map(lambda g: isinstance(g, str) and f"{g}_processed" in posedirs)

nvid = len(vids)
print(f"manifest videos: {nvid}  (unique rec-ids: {vids.rec.nunique()})")
print(f"  found in crosswalk:      {vids.in_xwalk.sum():5d} ({vids.in_xwalk.mean():.1%})")
print(f"  have superseded_gcp name:{vids.gcp.notna().sum():5d} ({vids.gcp.notna().mean():.1%})")
print(f"  POSE DIR ON DISK:        {vids.pose_on_disk.sum():5d} ({vids.pose_on_disk.mean():.1%})")
print("\nrelease breakdown of manifest videos:")
print(vids.release.value_counts(dropna=False).to_string())
print("\ndataset breakdown:")
print(vids.dataset.value_counts(dropna=False).to_string())
print(f"\nIMU processed (of manifest videos): {(vids.imu=='✅').sum()} ({(vids.imu=='✅').mean():.1%})")

# pair-level coverage (weight by how many training pairs each video contributes)
pose_vids = set(vids.loc[vids.pose_on_disk, "video_id"])
pair_cov = man.video_id.isin(pose_vids).mean()
print(f"\nPAIR-LEVEL pose coverage: {pair_cov:.1%} of the 140k training pairs")
print(f"total pose dirs on disk: {len(posedirs)}")
