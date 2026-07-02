"""Within-child train/eval (Vong-2026-style): train on a single child's OWN data, evaluate on
held-out videos of the SAME child. Holdout is by VIDEO (frames within a video are contiguous ->
frame-level holdout would leak), and temporal (test = latest 20% of videos by date). Builds:
  {CH}_train.parquet     : all train-video utterance pairs (region-embedded)
  eval_frames_{CH}.parquet: within-child 4AFC eval from CDI detections on TEST videos
"""
import sys, os
import pandas as pd, numpy as np
BV = "/ccn2a/dataset/babyview/2025.2/outputs"; W = "/data2/mcfrank/vlm-headcam"
CH = sys.argv[1] if len(sys.argv) > 1 else "S00510002"
TEST_FRAC = 0.2; MIN_CAT = 20; CAP = 300

fc = pd.read_csv(f"{BV}/full_clip_results.csv", usecols=[
    "child_id", "video_name", "utterance", "utterance_start_time", "utterance_end_time", "clip_score_max"]).dropna(
    subset=["clip_score_max", "utterance_start_time", "utterance_end_time"]).rename(columns={"video_name": "video_id", "utterance": "text"})
fc = fc[fc.child_id == CH].copy()
fc["frame_idx"] = ((fc.utterance_start_time + fc.utterance_end_time) / 2).astype(int)
ridx = pd.read_parquet(f"{W}/emb_reg/index.parquet")
have = set(zip(ridx.video_id, ridx.frame_idx.astype(int)))
fc = fc[[(v, int(f)) in have for v, f in zip(fc.video_id, fc.frame_idx)]]

# temporal video-level split: videos sorted by name (embeds date) -> last TEST_FRAC held out
vids = sorted(fc.video_id.unique())               # name = S..._YYYY-MM-DD_... => chronological
n_test = max(1, int(TEST_FRAC * len(vids)))
test_vids = set(vids[-n_test:]); train_vids = set(vids[:-n_test])
train = fc[fc.video_id.isin(train_vids)]
print(f"{CH}: {len(vids)} videos -> {len(train_vids)} train / {len(test_vids)} test")
print(f"  train pairs: {len(train)} (aligned {int((train.clip_score_max>0.24).sum())})")

train[["video_id", "frame_idx", "text", "clip_score_max", "child_id"]].reset_index(drop=True).to_parquet(f"{W}/manifests/{CH}_train.parquet")

# within-child eval from TEST videos' CDI detections (region-embedded frames only)
cdidir = f"{BV}/object_detections/cdi"
cats = {}
for v in test_vids:
    p = f"{cdidir}/{v}/bounding_box_predictions.csv"
    if not os.path.exists(p):
        continue
    dd = pd.read_csv(p).dropna(subset=["xmin"])
    for cls, g in dd.groupby("class_name"):
        for fn in g.frame_number.unique():
            if (v, int(fn)) in have:
                cats.setdefault(str(cls), []).append((v, int(fn)))
rows = []
rng = np.random.default_rng(0)
for cls, lst in cats.items():
    if len(lst) >= MIN_CAT:
        sel = lst if len(lst) <= CAP else [lst[i] for i in rng.choice(len(lst), CAP, replace=False)]
        for v, f in sel:
            rows.append(dict(video_id=v, frame_idx=int(f), category=cls))
ev = pd.DataFrame(rows)
ev.to_parquet(f"{W}/manifests/eval_frames_{CH}.parquet")
print(f"  within-child eval: {ev.category.nunique()} categories, {len(ev)} frames (from test videos)")
print("  top cats:", ev.category.value_counts().head(10).to_dict())
