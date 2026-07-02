"""Track A — social-cue audit.

Question: does any machine-readable social cue in the BabyView stream carry alignment
information that the endogenous bootstrap signals lacked (all had rho<=0.1 vs held-out CLIP)?

For each frame-utterance pair we build candidate cues and report:
  - Spearman rho(cue, clip_score_max)              [same yardstick that diagnosed E0-E9]
  - precision@top-k vs the clip>0.24 aligned base rate
  - a cross-validated logistic combination of all cues

Cues:
  detections (YOLOE 'cdi' set, ~38% of pairs): hands present/count, hand-object contact,
      person presence & apparent closeness (box area), scene object count.
  stability (all pairs): temporal cosine of adjacent 1fps DINOv2 CLS embeddings around the
      utterance frame -- a still world (a held/attended object) vs head-swinging locomotion.
  prosody (video-sampled, separate pass): energy & pitch dynamics over the utterance window
      -- a child-directed-speech emphasis proxy.

Usage:
  python cue_audit.py det_stab   --out FEATURES.parquet     # pass 1 (all pairs)
  python cue_audit.py prosody    --sample-videos 900 --out PROS.parquet   # pass 2
  python cue_audit.py report     --feat FEATURES.parquet [--pros PROS.parquet]
"""
import sys, os, subprocess, argparse
import numpy as np, pandas as pd
from pathlib import Path

BV = Path("/ccn2a/dataset/babyview/2025.2")
DETS = BV / "outputs/object_detections/cdi"
CLIP = BV / "outputs/full_clip_results.csv"
MP3 = BV / "mp3"
WORK = Path("/data2/mcfrank/vlm-headcam")
MAN = WORK / "manifests/boot_across_140000.parquet"
EMB_FULL = WORK / "emb_full"


# ---------- shared metrics ----------
def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 50:
        return np.nan, int(m.sum())
    ra = pd.Series(a[m]).rank().to_numpy(); rb = pd.Series(b[m]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0, int(m.sum())
    return float(np.corrcoef(ra, rb)[0, 1]), int(m.sum())


def precision_at_k(cue, clip, k_frac=0.1, thr=0.24):
    m = ~(np.isnan(cue) | np.isnan(clip))
    cue, clip = cue[m], clip[m]
    if len(cue) < 50:
        return np.nan
    k = max(1, int(k_frac * len(cue)))
    top = np.argsort(-cue)[:k]
    return float((clip[top] > thr).mean())


# ---------- pass 1: detections + stability ----------
def _iou_touch(b1, b2, gap):
    # overlap OR near-touch within `gap` px on both axes
    ax0, ay0, ax1, ay1 = b1; bx0, by0, bx1, by1 = b2
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)   # horizontal gap (0 if overlap)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return dx <= gap and dy <= gap


def det_features_for_video(vid, frames):
    """Return dict frame_idx -> feature dict, for the requested frame_idxs in one video."""
    p = DETS / vid / "bounding_box_predictions.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[d.frame_number.isin(frames)]
    if len(d) == 0:
        return {}
    # per-video frame size estimate from box extents (consistent within a video)
    fw = float(np.nanmax(d.xmax)) if d.xmax.notna().any() else np.nan
    fh = float(np.nanmax(d.ymax)) if d.ymax.notna().any() else np.nan
    area = fw * fh if fw and fh else np.nan
    out = {}
    for fi, g in d.groupby("frame_number"):
        g = g.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
        cls = g.class_name.astype(str)
        hands = g[cls == "hand"]; persons = g[cls == "person"]
        objs = g[~cls.isin(["hand", "person"])]
        pareas = ((persons.xmax - persons.xmin) * (persons.ymax - persons.ymin))
        # hand-object contact
        contact = 0
        if len(hands) and len(objs) and area == area:
            gap = 0.03 * (fw if fw == fw else 640)
            hb = hands[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
            ob = objs[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
            for h in hb:
                if any(_iou_touch(h, o, gap) for o in ob):
                    contact = 1; break
        out[int(fi)] = dict(
            hand_count=len(hands), hand_present=int(len(hands) > 0),
            hand_conf=float(hands.confidence.max()) if len(hands) else 0.0,
            person_count=len(persons), person_present=int(len(persons) > 0),
            person_area=float(pareas.max() / area) if len(persons) and area == area else 0.0,
            n_obj=int(objs.class_name.nunique()), n_det=len(g),
            hand_obj_contact=contact,
        )
    return out


def run_det_stab(out_path):
    man = pd.read_parquet(MAN)
    man["pid"] = np.arange(len(man))
    # ---- detections ----
    feats = {}
    groups = list(man.groupby("video_id"))
    for j, (vid, sub) in enumerate(groups):
        fm = det_features_for_video(vid, set(sub.frame_idx.astype(int)))
        if fm:
            for r in sub.itertuples(index=False):
                if int(r.frame_idx) in fm:
                    feats[r.pid] = fm[int(r.frame_idx)]
        if j % 500 == 0:
            print(f"  det {j}/{len(groups)} vids, {len(feats)} pairs covered", flush=True)
    det_df = pd.DataFrame.from_dict(feats, orient="index")
    det_cols = list(det_df.columns)
    print(f"detection coverage: {len(det_df)}/{len(man)} = {len(det_df)/len(man):.1%}", flush=True)

    # ---- stability from emb_full ----
    idx = pd.read_parquet(EMB_FULL / "index.parquet")
    emb = np.load(EMB_FULL / "emb.f16.npy", mmap_mode="r")
    lut = {(v, int(f)): int(r) for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row)}
    stab = np.full(len(man), np.nan, np.float32)

    def cos(r1, r2):
        a = np.asarray(emb[r1], np.float32); b = np.asarray(emb[r2], np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else np.nan

    for r in man.itertuples(index=False):
        v, fi = r.video_id, int(r.frame_idx)
        c = lut.get((v, fi))
        if c is None:
            continue
        vals = []
        for dk in (-2, -1, 1, 2):
            n = lut.get((v, fi + dk))
            if n is not None:
                vals.append(cos(c, n))
        if vals:
            stab[r.pid] = np.mean(vals)
    print(f"stability coverage: {np.isfinite(stab).mean():.1%}", flush=True)

    # ---- assemble ----
    df = man[["pid", "video_id", "frame_idx", "child_id", "clip_score_max"]].copy()
    df = df.merge(det_df.reset_index().rename(columns={"index": "pid"}), on="pid", how="left")
    df["stability"] = stab
    df["has_det"] = df["hand_present"].notna().astype(int)
    df.to_parquet(out_path)
    print("wrote", out_path, "cols:", [c for c in df.columns if c not in ("pid","video_id","frame_idx","child_id")])


# ---------- pass 2: prosody ----------
def load_wav_ffmpeg(mp3_path, sr=16000):
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(mp3_path), "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    except Exception:
        return None
    if not raw:
        return None
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def prosody_window(x, sr, t0, t1):
    a, b = int(t0 * sr), int(t1 * sr)
    seg = x[max(0, a):b]
    if len(seg) < sr // 5:            # <0.2s, skip
        return None
    fl = int(0.025 * sr); hop = int(0.010 * sr)
    frames = [seg[i:i + fl] for i in range(0, len(seg) - fl, hop)]
    if len(frames) < 3:
        return None
    F = np.stack(frames)
    rms = np.sqrt((F ** 2).mean(1) + 1e-9)
    # spectral centroid
    win = np.hanning(fl)
    mag = np.abs(np.fft.rfft(F * win, axis=1))
    freqs = np.fft.rfftfreq(fl, 1 / sr)
    cent = (mag * freqs).sum(1) / (mag.sum(1) + 1e-9)
    # crude autocorr pitch on voiced (high-energy) frames
    voiced = rms > np.median(rms)
    f0 = []
    lo, hi = int(sr / 400), int(sr / 75)     # 75-400 Hz
    for fr in F[voiced]:
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        if len(ac) <= hi or ac[0] <= 0:
            continue
        lag = lo + np.argmax(ac[lo:hi])
        if ac[lag] > 0.3 * ac[0]:
            f0.append(sr / lag)
    f0 = np.array(f0)
    return dict(
        rms_mean=float(rms.mean()), rms_range=float(rms.max() - rms.min()),
        rms_cv=float(rms.std() / (rms.mean() + 1e-9)),
        cent_mean=float(cent.mean()), cent_std=float(cent.std()),
        f0_mean=float(f0.mean()) if len(f0) else np.nan,
        f0_range=float(f0.max() - f0.min()) if len(f0) > 1 else np.nan,
        f0_std=float(f0.std()) if len(f0) > 1 else np.nan,
        dur=float(t1 - t0),
    )


def run_prosody(n_videos, out_path, seed=0):
    cr = pd.read_csv(CLIP, usecols=["child_id", "video_name", "utterance",
                                    "utterance_start_time", "utterance_end_time", "clip_score_max"])
    cr = cr.rename(columns={"video_name": "video_id"}).dropna(
        subset=["clip_score_max", "utterance_start_time", "utterance_end_time"])
    # restrict to videos in the training manifest, sample some
    man_vids = set(pd.read_parquet(MAN).video_id.unique())
    cr = cr[cr.video_id.isin(man_vids)]
    rng = np.random.default_rng(seed)
    vids = rng.choice(cr.video_id.unique(), min(n_videos, cr.video_id.nunique()), replace=False)
    rows = []
    for j, vid in enumerate(vids):
        child = vid.split("_")[0]
        mp = MP3 / child / f"{vid}.mp3"
        if not mp.exists():
            continue
        x = load_wav_ffmpeg(mp)
        if x is None:
            continue
        sr = 16000
        for r in cr[cr.video_id == vid].itertuples(index=False):
            pf = prosody_window(x, sr, r.utterance_start_time, r.utterance_end_time)
            if pf:
                pf.update(video_id=vid, clip_score_max=r.clip_score_max)
                rows.append(pf)
        if j % 50 == 0:
            print(f"  prosody {j}/{len(vids)} vids, {len(rows)} utts", flush=True)
    pd.DataFrame(rows).to_parquet(out_path)
    print("wrote", out_path, len(rows), "utterances")


# ---------- report ----------
def logistic_cv(X, y, folds=5, seed=0):
    """Tiny L2 logistic regression, k-fold; return mean held-out AUC & Spearman of scores."""
    from itertools import product
    n = len(y); rng = np.random.default_rng(seed); order = rng.permutation(n)
    Xn = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)
    Xn = np.nan_to_num(Xn)
    Xn = np.hstack([Xn, np.ones((n, 1))])
    aucs, scores = [], np.full(n, np.nan)
    for f in range(folds):
        te = order[f::folds]; tr = np.setdiff1d(order, te)
        w = np.zeros(Xn.shape[1])
        for _ in range(300):
            p = 1 / (1 + np.exp(-Xn[tr] @ w))
            g = Xn[tr].T @ (p - y[tr]) / len(tr) + 1e-2 * w
            w -= 0.5 * g
        s = Xn[te] @ w; scores[te] = s
        # AUC
        pos, neg = s[y[te] == 1], s[y[te] == 0]
        if len(pos) and len(neg):
            aucs.append(float((pos[:, None] > neg[None, :]).mean()))
    return np.mean(aucs), scores


def report(feat_path, pros_path=None):
    df = pd.read_parquet(feat_path)
    clip = df.clip_score_max.to_numpy()
    print("\n=== TRACK A: social-cue audit ===")
    print(f"pairs={len(df)}  aligned base rate (clip>0.24)={ (clip>0.24).mean():.1%}")
    print(f"detection coverage={df.has_det.mean():.1%}  stability coverage={df.stability.notna().mean():.1%}\n")
    cues = ["hand_present", "hand_count", "hand_conf", "hand_obj_contact",
            "person_present", "person_area", "person_count", "n_obj", "n_det", "stability"]
    print(f"{'cue':18s} {'rho':>7s} {'n':>8s} {'prec@10%':>9s}  (base rate below)")
    rows = []
    for c in cues:
        if c not in df:
            continue
        x = df[c].to_numpy(float)
        rho, n = spearman(x, clip); p10 = precision_at_k(x, clip, 0.1)
        rows.append((c, rho, n, p10)); print(f"{c:18s} {rho:7.3f} {n:8d} {p10:9.3f}")
    base = (clip > 0.24).mean()
    print(f"{'(base rate)':18s} {'':7s} {'':8s} {base:9.3f}")

    # combined logistic on detection subset (where all det cues present)
    sub = df[df.has_det == 1].copy()
    detc = ["hand_present", "hand_count", "hand_obj_contact", "person_area", "n_obj", "stability"]
    detc = [c for c in detc if c in sub]
    X = sub[detc].to_numpy(float); y = (sub.clip_score_max.to_numpy() > 0.24).astype(float)
    auc, sc = logistic_cv(X, y)
    rho_c, _ = spearman(sc, sub.clip_score_max.to_numpy())
    print(f"\ncombined logistic (det subset, {len(sub)} pairs, cues={detc}):")
    print(f"  CV AUC={auc:.3f}  rho(score,clip)={rho_c:.3f}  prec@10%={precision_at_k(sc, sub.clip_score_max.to_numpy(),0.1):.3f}")

    if pros_path and os.path.exists(pros_path):
        pr = pd.read_parquet(pros_path)
        pclip = pr.clip_score_max.to_numpy()
        print(f"\n--- prosody ({len(pr)} utterances, {pr.video_id.nunique()} videos) ---")
        pcues = ["rms_range", "rms_cv", "cent_std", "f0_range", "f0_std", "f0_mean", "dur"]
        for c in pcues:
            if c in pr:
                rho, n = spearman(pr[c].to_numpy(float), pclip)
                print(f"{c:18s} {rho:7.3f} {n:8d} {precision_at_k(pr[c].to_numpy(float), pclip,0.1):9.3f}")
        Xp = pr[[c for c in pcues if c in pr]].to_numpy(float)
        yp = (pclip > 0.24).astype(float)
        aucp, scp = logistic_cv(Xp, yp)
        print(f"combined prosody logistic: CV AUC={aucp:.3f}  rho={spearman(scp,pclip)[0]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["det_stab", "prosody", "report"])
    ap.add_argument("--out", default=str(WORK / "runs/cue_feats.parquet"))
    ap.add_argument("--feat", default=str(WORK / "runs/cue_feats.parquet"))
    ap.add_argument("--pros", default=str(WORK / "runs/cue_prosody.parquet"))
    ap.add_argument("--sample-videos", type=int, default=900)
    a = ap.parse_args()
    if a.cmd == "det_stab":
        run_det_stab(a.out)
    elif a.cmd == "prosody":
        run_prosody(a.sample_videos, a.out)
    else:
        report(a.feat, a.pros)
