"""Exp D region grounding. For aligned utterances (clip>0.24) in videos that have YOLOE
detections, pick the single dominant detected object in the utterance window and record
its frame + bbox. Emits a manifest where the visual side is an OBJECT CROP, not the scene.

Also emits a matched whole-frame manifest over the SAME utterances (video_id, frame_idx =
the detection frame) so 'crop vs frame' is the only difference between the two arms.

Outputs:
  manifests/region_crop_<held>.parquet   video_id, frame_idx, xmin,ymin,xmax,ymax, text
  manifests/region_frame_<held>.parquet  video_id, frame_idx, text   (same utterances)
"""
import argparse
import os
import pandas as pd
from pathlib import Path

from common import CLIP_RESULTS, DETS, MANIFEST_DIR, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="S00360001")
    ap.add_argument("--min-clip", type=float, default=0.24)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-pix", type=int, default=2000)
    args = ap.parse_args()

    det_videos = set(os.listdir(DETS))
    cr = pd.read_csv(CLIP_RESULTS, usecols=[
        "child_id", "video_name", "utterance", "utterance_start_time",
        "utterance_end_time", "clip_score_max"]).dropna()
    cr = cr[(cr.clip_score_max >= args.min_clip) & (cr.child_id.astype(str) != args.held)]
    cr = cr[cr.video_name.isin(det_videos)]
    cr = cr[cr.utterance.map(lambda t: len(tokenize(t)) >= 1)]
    print(f"aligned utterances in detection-videos (excl {args.held}): {len(cr)}", flush=True)

    crop_rows, frame_rows = [], []
    for vid, g in cr.groupby("video_name"):
        csv = DETS / vid / "bounding_box_predictions.csv"
        if not csv.exists():
            continue
        det = pd.read_csv(csv, usecols=["frame_number", "xmin", "ymin", "xmax", "ymax",
                                        "confidence", "masked_pixel_count"]).dropna()
        det = det[(det.confidence >= args.conf) & (det.masked_pixel_count >= args.min_pix)]
        if det.empty:
            continue
        for r in g.itertuples(index=False):
            lo, hi = int(r.utterance_start_time), int(r.utterance_end_time)
            win = det[(det.frame_number >= lo) & (det.frame_number <= hi)]
            if win.empty:
                continue
            top = win.loc[win.masked_pixel_count.idxmax()]
            fidx = int(top.frame_number)
            crop_rows.append((vid, fidx, float(top.xmin), float(top.ymin),
                              float(top.xmax), float(top.ymax), r.utterance))
            frame_rows.append((vid, fidx, r.utterance))

    crop = pd.DataFrame(crop_rows, columns=["video_id", "frame_idx", "xmin", "ymin",
                                            "xmax", "ymax", "text"])
    frame = pd.DataFrame(frame_rows, columns=["video_id", "frame_idx", "text"])
    Path(MANIFEST_DIR).mkdir(parents=True, exist_ok=True)
    crop.to_parquet(Path(MANIFEST_DIR) / f"region_crop_{args.held}.parquet")
    frame.to_parquet(Path(MANIFEST_DIR) / f"region_frame_{args.held}.parquet")
    print(f"region pairs: {len(crop)} (crop + matched frame manifests written)")


if __name__ == "__main__":
    main()
