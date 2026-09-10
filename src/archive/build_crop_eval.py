"""Build a crop-based eval: for each labeled eval frame, recover the dominant detection's
bbox for that category so the eval can be run on object crops (matched to crop-trained
models). Output has video_id, frame_idx, xmin..ymax, category  -> feed to embed_crops.py."""
import argparse
import pandas as pd
from pathlib import Path
from common import DETS, MANIFEST_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-frames", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ev = pd.read_parquet(args.eval_frames)
    rows = []
    for vid, g in ev.groupby("video_id"):
        csv = DETS / vid / "bounding_box_predictions.csv"
        if not csv.exists():
            continue
        det = pd.read_csv(csv, usecols=["frame_number", "xmin", "ymin", "xmax", "ymax",
                                        "class_name", "masked_pixel_count"]).dropna()
        det["class_name"] = det.class_name.str.lower().str.strip()
        for r in g.itertuples(index=False):
            d = det[(det.frame_number == r.frame_idx) & (det.class_name == r.category)]
            if d.empty:
                continue
            top = d.loc[d.masked_pixel_count.idxmax()]
            rows.append((vid, int(r.frame_idx), float(top.xmin), float(top.ymin),
                         float(top.xmax), float(top.ymax), r.category))
    out = pd.DataFrame(rows, columns=["video_id", "frame_idx", "xmin", "ymin",
                                      "xmax", "ymax", "category"])
    out.to_parquet(args.out)
    print(f"crop-eval: {len(out)} frames / {out.category.nunique()} cats -> {args.out}")


if __name__ == "__main__":
    main()
