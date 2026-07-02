"""Frame-center cue = FTF's 'child's eyes' (F=.54, the best single cue): in a headcam the
child's gaze IS the frame center. Proxy: how distinct is the central region from the
periphery in DINOv2 space (a foregrounded, attended object at center vs uniform wall/floor).
emb_reg = [N, 1+16, 768], row0=CLS, grid row-major 4x4; center 2x2 = emb idx 6,7,10,11."""
import numpy as np, pandas as pd
W = "/data2/mcfrank/vlm-headcam"
idx = pd.read_parquet(f"{W}/emb_reg/index.parquet")
emb = np.load(f"{W}/emb_reg/emb.f16.npy", mmap_mode="r")   # [N,17,768]
CENTER = [6, 7, 10, 11]
PERIPH = [i for i in range(1, 17) if i not in CENTER]
man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")[["video_id", "frame_idx"]]
lut = {(v, int(f)): int(r) for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row)}

def cosd(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)

rows = []
for r in man.itertuples(index=False):
    row = lut.get((r.video_id, int(r.frame_idx)))
    if row is None:
        continue
    g = np.asarray(emb[row], np.float32)          # [17,768]
    c = g[CENTER].mean(0); p = g[PERIPH].mean(0)
    rows.append(dict(video_id=r.video_id, frame_idx=int(r.frame_idx),
                     center_distinct=1.0 - cosd(c, p),
                     center_norm=float(np.linalg.norm(c - g[1:].mean(0)))))
df = pd.DataFrame(rows)
df.to_parquet(f"{W}/manifests/frame_center.parquet")
# quick screen
clip = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet").merge(
    df, on=["video_id", "frame_idx"]).clip_score_max.to_numpy()
def sp(x):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(clip).rank())[0, 1])
print(f"frame_center: {len(df)} pairs")
print(f"  rho(center_distinct, clip) = {sp(df.center_distinct.to_numpy()):+.3f}")
print(f"  rho(center_norm, clip)     = {sp(df.center_norm.to_numpy()):+.3f}")
