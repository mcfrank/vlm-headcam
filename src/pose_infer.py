"""Whole-body (133-kpt COCO-WholeBody) pose over 2025.2 frames via rtmlib (RTMDet + RTMW,
ONNX/CUDA) — no mmcv/mmdet needed. Processes frames in their native 2025.2 naming, so no
gcp-hash crosswalk. Per video, writes one compact .npz (ragged persons flattened).

Needs the CUDA-12 onnxruntime to find cuDNN 9 — run with:
  NV=/data2/mcfrank/ladder/condaenv/lib/python3.11/site-packages/nvidia
  export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES=<gpu> pose_env/bin/python src/pose_infer.py ...

Modes:
  --benchmark VIDEO_ID   time throughput on one video + save skeleton overlays for review
  --shard FILE           process the video_ids listed in FILE (one per line), write .npz each
"""
import argparse
import time
from pathlib import Path
import numpy as np
import cv2

FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")


def make_model(mode):
    from rtmlib import Wholebody
    return Wholebody(mode=mode, backend="onnxruntime", device="cuda")


def frames_of(video_id):
    d = FRAMES / video_id
    return sorted(d.glob("*.jpg"))


def run_video(model, video_id, limit=0):
    """Return per-person flattened arrays + frame bookkeeping for one video."""
    fps_paths = frames_of(video_id)
    if limit:
        fps_paths = fps_paths[:limit]
    fidx, npers = [], []
    kp_all, sc_all, box_all, pf_all = [], [], [], []
    for p in fps_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        fi = int(p.stem)
        kps, scs = model(img)                       # [P,133,2], [P,133]
        fidx.append(fi); npers.append(len(kps))
        for j in range(len(kps)):
            kp_all.append(kps[j]); sc_all.append(scs[j]); pf_all.append(fi)
    out = dict(
        frame_idx=np.asarray(fidx, np.int32),
        n_person=np.asarray(npers, np.int16),
        kp=(np.asarray(kp_all, np.float16) if kp_all else np.zeros((0, 133, 2), np.float16)),
        sc=(np.asarray(sc_all, np.float16) if sc_all else np.zeros((0, 133), np.float16)),
        person_frame=np.asarray(pf_all, np.int32),
    )
    return out, len(fidx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="performance", choices=["performance", "balanced", "lightweight"])
    ap.add_argument("--benchmark", metavar="VIDEO_ID")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--shard")
    ap.add_argument("--outdir", default="/data2/mcfrank/pose_2025_2")
    ap.add_argument("--overlays", default="/data2/mcfrank/vlm-headcam/pose_bench_overlays")
    args = ap.parse_args()

    model = make_model(args.mode)

    if args.benchmark:
        vid = args.benchmark
        # warmup
        w = frames_of(vid)[:5]
        for p in w:
            im = cv2.imread(str(p))
            if im is not None:
                model(im)
        t0 = time.time()
        out, nf = run_video(model, vid, limit=args.limit)
        dt = time.time() - t0
        cov = float(np.mean(out["n_person"] > 0)) if nf else 0.0
        ppf = float(out["n_person"].mean()) if nf else 0.0
        print(f"BENCHMARK {vid}")
        print(f"  frames={nf}  time={dt:.1f}s  throughput={nf/dt:.1f} fps")
        print(f"  coverage(>=1 person)={cov*100:.1f}%  mean persons/frame={ppf:.2f}")
        # overlays for human review (faces present — reviewer looks, not us)
        from rtmlib import draw_skeleton
        od = Path(args.overlays); od.mkdir(parents=True, exist_ok=True)
        saved, i = 0, 0
        paths = frames_of(vid)
        while saved < 8 and i < len(paths):
            im = cv2.imread(str(paths[i]))
            if im is not None:
                kps, scs = model(im)
                if len(kps):
                    vis = draw_skeleton(im.copy(), kps, scs, kpt_thr=0.4)
                    cv2.imwrite(str(od / f"{vid}_{paths[i].stem}.jpg"), vis)
                    saved += 1
            i += 1
        print(f"  wrote {saved} overlays -> {od}")
        return

    if args.shard:
        vids = [l.strip() for l in open(args.shard) if l.strip()]
        outd = Path(args.outdir); outd.mkdir(parents=True, exist_ok=True)
        t0, done = time.time(), 0
        for vid in vids:
            fp = outd / f"{vid}.npz"
            if fp.exists():
                continue
            out, nf = run_video(model, vid)
            np.savez_compressed(fp, **out)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(vids)} videos  {(time.time()-t0)/done:.1f}s/vid", flush=True)
        print(f"DONE shard: {done} videos in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
