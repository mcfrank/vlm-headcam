"""Old (2025.03 mmpose) vs new (RTMW/rtmlib) pose on identical frames, drawn with ONE
clean drawer (limbs as lines; face + hands as dots — no dense face-contour scribble).
Side-by-side panels for human review. Runs on the old sampled_frames so the pixels match.
"""
import argparse, sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
import pose_lib
from pose_infer import make_model, infer

BODY_EDGES = pose_lib.BODY_EDGES
FACE, LHAND, RHAND = slice(23, 91), slice(91, 112), slice(112, 133)


def draw(img, persons, thr):
    """persons: list of (kp[133,2], score[133]). Body limbs as lines; joints/face/hands as dots."""
    for kp, sc in persons:
        for a, b in BODY_EDGES:
            if sc[a] > thr and sc[b] > thr:
                cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (0, 255, 0), 2)
        for i in range(17):                       # body joints
            if sc[i] > thr:
                cv2.circle(img, tuple(kp[i].astype(int)), 3, (0, 255, 255), -1)
        for i in range(23, 91):                   # face
            if sc[i] > thr:
                cv2.circle(img, tuple(kp[i].astype(int)), 1, (255, 200, 0), -1)
        for i in range(91, 133):                  # hands
            if sc[i] > thr:
                cv2.circle(img, tuple(kp[i].astype(int)), 2, (255, 0, 255), -1)
    return img


def label(img, txt):
    cv2.rectangle(img, (0, 0), (len(txt) * 12 + 12, 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcp", default="00220001_2024-02-05_1_57f12f7764_processed")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--old-thr", type=float, default=0.3)
    ap.add_argument("--new-thr", type=float, default=1.0)
    ap.add_argument("--out", default="/data2/mcfrank/vlm-headcam/pose_compare")
    args = ap.parse_args()

    pose_dir = Path(pose_lib.POSE_ROOT) / args.gcp
    frame_dir = Path(pose_lib.FRAME_ROOT) / args.gcp
    secs = sorted(int(p.stem) for p in pose_dir.glob("*.pkl"))
    if not secs:
        print("no old pkls for", args.gcp); return
    pick = secs[:: max(1, len(secs) // args.n)][: args.n]

    model = make_model("performance")
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    for sec in pick:
        fp = frame_dir / f"{sec:05d}.jpg"
        img = cv2.imread(str(fp))
        if img is None:
            continue
        # old
        try:
            op = pose_lib.load_pose(str(pose_dir / f"{sec:05d}.pkl"))
            old_persons = [(P["kp"], P["score"]) for P in op["persons"]]
        except Exception as e:
            old_persons = []
        old_img = label(draw(img.copy(), old_persons, args.old_thr), f"OLD mmpose  n={len(old_persons)}")
        # new
        kps, scs = infer(model, img)
        new_persons = list(zip(np.asarray(kps, np.float32), np.asarray(scs, np.float32)))
        new_img = label(draw(img.copy(), new_persons, args.new_thr), f"NEW RTMW  n={len(new_persons)}")
        panel = np.hstack([old_img, new_img])
        cv2.imwrite(str(od / f"{args.gcp}_{sec:05d}.jpg"), panel)
    print(f"wrote {len(pick)} old|new panels -> {od}")


if __name__ == "__main__":
    main()
