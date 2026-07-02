"""Build a CVCL-style CDI-category eval set from YOLOE detections on held-out videos.

A frame becomes a labeled example of category C when C is the *dominant* CDI object in
that frame (largest by masked pixels, confident, and a clear majority of detected object
area) -- i.e. the frame is unambiguously "about" C. Trials (4AFC) are generated later, at
eval time, from these labeled frames.

Output: MANIFEST_DIR/eval_frames.parquet  columns video_id, frame_idx, category
"""
import argparse
import pandas as pd
from pathlib import Path

from common import DETS, MANIFEST_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True,
                    help="text file: held-out video_ids (one per line) to draw eval frames from")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-pix", type=int, default=4000, help="min masked pixels for dominant obj")
    ap.add_argument("--dominance", type=float, default=0.5,
                    help="dominant obj must be >= this fraction of total detected object area")
    ap.add_argument("--min-frames", type=int, default=20, help="min labeled frames to keep a category")
    ap.add_argument("--max-per-cat", type=int, default=300, help="cap frames per category")
    ap.add_argument("--out", default=str(MANIFEST_DIR / "eval_frames.parquet"))
    args = ap.parse_args()

    videos = [v.strip() for v in open(args.videos) if v.strip()]
    labeled = []  # (video_id, frame_idx, category)

    for vid in videos:
        csv = DETS / vid / "bounding_box_predictions.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, usecols=["frame_number", "confidence", "class_name",
                                       "masked_pixel_count"])
        df = df.dropna(subset=["confidence", "masked_pixel_count", "class_name"])
        df = df[df.confidence >= args.conf]
        if df.empty:
            continue
        # per frame: total object area, and the single largest object
        for fnum, g in df.groupby("frame_number"):
            tot = g.masked_pixel_count.sum()
            top = g.loc[g.masked_pixel_count.idxmax()]
            if top.masked_pixel_count < args.min_pix:
                continue
            if tot > 0 and (top.masked_pixel_count / tot) < args.dominance:
                continue
            labeled.append((vid, int(fnum), str(top.class_name).lower().strip()))

    lab = pd.DataFrame(labeled, columns=["video_id", "frame_idx", "category"])
    # keep categories with enough clean frames; cap per category
    counts = lab.category.value_counts()
    keep = counts[counts >= args.min_frames].index
    lab = lab[lab.category.isin(keep)]
    lab = (lab.groupby("category", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), args.max_per_cat), random_state=0)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lab.reset_index(drop=True).to_parquet(args.out)
    print(f"eval: {len(lab)} labeled frames across {lab.category.nunique()} categories")
    print(lab.category.value_counts().to_string())


if __name__ == "__main__":
    main()
