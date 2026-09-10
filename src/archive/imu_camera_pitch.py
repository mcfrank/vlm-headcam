#!/usr/bin/env python3
"""Camera pitch from the GoPro IMU sidecars (*_metadata_imu.zip / imu_combined.csv), per video, by child x camera.

Estimator: median unit specific-force vector a (60 Hz accelerometer; median over the whole recording is robust to
head motion). The IMU Z axis is the optical axis (forward/back); X/Y span the sensor plane and get swapped/flipped by
landscape (`_rotated`) or upside-down mountings, Z does not. So
    pitch_down_deg = asin(a_z)      (+ = optical axis BELOW horizontal. Sign calibrated empirically on 2025.2 bones
                                     videos: larger values -> caregiver faces higher in frame (r=-0.45), more of the
                                     child's own hands in view (r=+0.65), and the known V2->V3 mount change.)
    tilt_from_dominant_axis = angle between a and the dominant sensor-plane axis (sanity / roll)
GoPro's own fused gravity stream (GRAV) exists on bones only -> used to validate the accelerometer estimator.

  extract:  python imu_camera_pitch.py extract --pull_root <.../gcloud/pull> --out imu_video_pitch.tsv [--procs 12]
  analyze:  python imu_camera_pitch.py analyze --imu imu_video_pitch.tsv --crosswalk <tsv> --out <dir>
                [--pose_video_summary video_summary.csv]   # from pose_hand_camera.py, for the pitch-vs-hand-y check
Aggregates only; nothing identifying leaves the node beyond subject ids + dates already in the registry.
"""
import argparse, glob, io, json, os, time, zipfile
import numpy as np, pandas as pd
from multiprocessing import Pool

CAM_COLORS = {'bones': '#2a78d6', 'mini': '#eb6834', 'headlight': '#1baf7a'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e8e7e3', '#fcfcfb'


def one(zpath):
    vid = os.path.basename(zpath).replace('_metadata_imu.zip', '')
    rec = {'video_id_imu': vid, 'status': 'ok'}
    try:
        z = zipfile.ZipFile(zpath)
        if 'imu_combined.csv' not in z.namelist():
            rec['status'] = 'no_combined_csv'; return rec
        df = pd.read_csv(io.BytesIO(z.read('imu_combined.csv')))
        cols = {c.split(' ')[0]: c for c in df.columns}
        need = ['ACCL_X', 'ACCL_Y', 'ACCL_Z']
        if any(k not in cols for k in need):
            rec['status'] = 'no_accl'; return rec
        A = df[[cols[k] for k in need]].to_numpy(float); A = A[np.isfinite(A).all(1)]
        ts = df[cols['Timestamp']].to_numpy(float) if 'Timestamp' in cols else None
        rec['n'] = int(len(A)); rec['dur_s'] = float(np.nanmax(ts) - np.nanmin(ts)) if ts is not None and len(ts) else np.nan
        if len(A) < 100:
            rec['status'] = 'too_few'; return rec
        nrm = np.linalg.norm(A, axis=1); U = A / nrm[:, None]
        m = np.median(U, axis=0); m /= np.linalg.norm(m)
        rec.update(ax=m[0], ay=m[1], az=m[2], g_med=float(np.median(nrm)))
        rec['static_frac'] = float(((nrm > 8.8) & (nrm < 10.8)).mean())
        dom = int(np.argmax(np.abs(m[:2]))); rec['dominant_axis'] = 'XY'[dom]; rec['dominant_sign'] = int(np.sign(m[dom]))
        rec['pitch_down_deg'] = float(np.degrees(np.arcsin(np.clip(m[2], -1, 1))))
        ps = np.degrees(np.arcsin(np.clip(U[:, 2], -1, 1)))
        rec['pitch_q10'], rec['pitch_q50'], rec['pitch_q90'] = [float(x) for x in np.percentile(ps, [10, 50, 90])]
        rec['pitch_sd'] = float(np.std(ps))
        # roll-ish: angle of a within the sensor plane relative to the dominant axis
        rec['inplane_deg'] = float(np.degrees(np.arctan2(m[1 - dom], m[dom] * rec['dominant_sign'])))
        if all(k in cols for k in ['GRAV_X', 'GRAV_Y', 'GRAV_Z']):
            G = df[[cols[k] for k in ['GRAV_X', 'GRAV_Y', 'GRAV_Z']]].to_numpy(float); G = G[np.isfinite(G).all(1)]
            if len(G) > 100:
                g = np.median(G / np.linalg.norm(G, axis=1)[:, None], axis=0); g /= np.linalg.norm(g)
                rec.update(gx=g[0], gy=g[1], gz=g[2], grav_pitch_down_deg=float(np.degrees(np.arcsin(np.clip(g[2], -1, 1)))))
    except Exception as e:
        rec['status'] = 'err:' + type(e).__name__
    return rec


def extract(args):
    zips = sorted(glob.glob(os.path.join(args.pull_root, '**', '*_metadata_imu.zip'), recursive=True))
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, sep='\t').video_id_imu)
    todo = [z for z in zips if os.path.basename(z).replace('_metadata_imu.zip', '') not in done]
    print(f'{len(zips)} zips, {len(todo)} to do, {args.procs} procs', flush=True)
    t0 = time.time(); rows = []
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=4), 1):
            rows.append(r)
            if i % 500 == 0 or i == len(todo):
                pd.DataFrame(rows).to_csv(args.out, sep='\t', index=False, mode='a', header=not os.path.exists(args.out)); rows = []
                print(f'[{i}/{len(todo)}] {(time.time() - t0) / 60:.1f} min', flush=True)
    print('EXTRACT_DONE', flush=True)


def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=2, width=0.6); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.set_facecolor(SURF)


def analyze(args):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'text.color': INK, 'axes.labelcolor': INK2, 'figure.facecolor': SURF, 'savefig.facecolor': SURF})
    os.makedirs(args.out, exist_ok=True)
    imu = pd.read_csv(args.imu, sep='\t').rename(columns={'pitch_deg': 'pitch_down_deg', 'grav_pitch_deg': 'grav_pitch_down_deg'})
    print('imu rows', len(imu), imu.status.value_counts().to_dict())
    fov = {k: float(v) for k, v in (x.split('=') for x in args.fov)}
    xw = pd.read_csv(args.crosswalk, sep='\t', low_memory=False)
    # IMU zips are named by the bare recording id: strip the video variants (_rotated, _blackout, _processed)
    xw['video_id_imu'] = xw.video_id.str.replace(r'(_rotated|_blackout|_processed)', '', regex=True)
    xw['video_id'] = xw.video_id.str.replace('_processed', '', regex=False)   # match create_csv.py's ids
    xw['date'] = pd.to_datetime(xw.date, errors='coerce'); xw['rotated'] = xw.video_id.str.endswith('_rotated')
    d = imu[imu.status == 'ok'].merge(xw[['video_id', 'video_id_imu', 'subject_id', 'camera', 'date', 'duration_sec', 'rotated']], on='video_id_imu', how='left')
    d = d[d.camera.notna() & ~d.camera.isin(set(filter(None, args.exclude_cameras.split(','))))].copy()
    # field-of-view edges (deg below horizontal; vertical FOV per camera): bottom = pitch_down + FOV/2, top = FOV/2 - pitch_down
    d['half_fov'] = d.camera.map(fov); d['edge_bottom_deg'] = d.pitch_down_deg + d.half_fov / 2; d['edge_top_deg'] = d.half_fov / 2 - d.pitch_down_deg
    if args.release_ids:      # the pull is a superset of the release
        keep = {l.strip().replace('_processed', '') for l in open(args.release_ids) if l.strip()}
        n0 = len(d); d = d[d.video_id.isin(keep)]
        print(f'release filter: kept {len(d)} of {n0} videos')
    print('joined', len(d), d.camera.value_counts().to_dict())
    print('dominant axis x camera:'); print(pd.crosstab([d.camera, d.rotated], [d.dominant_axis, d.dominant_sign]))
    if 'grav_pitch_down_deg' in d:
        g = d.dropna(subset=['grav_pitch_down_deg'])
        if len(g): print(f'GRAV vs ACCL pitch (n={len(g)}): r={np.corrcoef(g.pitch_down_deg, g.grav_pitch_down_deg)[0, 1]:.3f}, mean abs diff={np.abs(g.pitch_down_deg - g.grav_pitch_down_deg).mean():.2f} deg')
    # optional: pose-based hand position per video -> sign + validity of the tilt indicator
    if args.pose_video_summary:
        pv = pd.read_csv(args.pose_video_summary)
        d = d.merge(pv[['video_id', 'p_ego_hand', 'ego_y_mean', 'face_y_mean', 'n_frames']], on='video_id', how='left')
        for c in sorted(d.camera.unique()):
            s = d[(d.camera == c) & d.ego_y_mean.notna() & (d.n_frames >= 300)]
            if len(s) > 20:
                print(f'{c}: n={len(s)} corr(pitch, ego_y_mean)={s.pitch_down_deg.corr(s.ego_y_mean):+.3f}  corr(pitch, face_y_mean)={s.pitch_down_deg.corr(s.face_y_mean):+.3f}  corr(pitch, p_ego_hand)={s.pitch_down_deg.corr(s.p_ego_hand):+.3f}')
    d.to_csv(f'{args.out}/imu_video_pitch_joined.csv', index=False)
    sc = d.groupby(['subject_id', 'camera']).agg(n_videos=('pitch_down_deg', 'size'), hours=('duration_sec', lambda s: s.sum() / 3600),
                                                 pitch_mean=('pitch_down_deg', 'mean'), pitch_median=('pitch_down_deg', 'median'), pitch_sd_between=('pitch_down_deg', 'std'),
                                                 pitch_sd_within=('pitch_sd', 'mean'), edge_bottom_mean=('edge_bottom_deg', 'mean'), edge_top_mean=('edge_top_deg', 'mean'),
                                                 first_date=('date', 'min'), last_date=('date', 'max')).reset_index()
    sc.to_csv(f'{args.out}/imu_subject_camera_summary.csv', index=False)
    if {'bones', 'mini'} <= set(sc.camera):
        w = sc[sc.hours >= args.min_hours_pair].pivot(index='subject_id', columns='camera', values=['hours', 'pitch_mean', 'edge_bottom_mean']).dropna()
        if len(w):
            tab = pd.DataFrame({'hours_bones': w[('hours', 'bones')], 'hours_mini': w[('hours', 'mini')], 'pitch_bones': w[('pitch_mean', 'bones')],
                                'pitch_mini': w[('pitch_mean', 'mini')], 'delta_pitch_mini_minus_bones': w[('pitch_mean', 'mini')] - w[('pitch_mean', 'bones')],
                                'edge_bottom_bones': w[('edge_bottom_mean', 'bones')], 'edge_bottom_mini': w[('edge_bottom_mean', 'mini')]})
            tab['delta_edge_bottom'] = tab.edge_bottom_mini - tab.edge_bottom_bones
            tab.round(1).to_csv(f'{args.out}/imu_within_child_table.csv')
            print(tab.round(1).to_string()); print('mean delta pitch', round(tab.delta_pitch_mini_minus_bones.mean(), 1), 'sd', round(tab.delta_pitch_mini_minus_bones.std(), 1),
                  '| mean delta bottom edge', round(tab.delta_edge_bottom.mean(), 1))
    pooled = d.groupby('camera').pitch_down_deg.describe().round(2); print(pooled); pooled.to_csv(f'{args.out}/imu_camera_summary.csv')
    cams = [c for c in ['bones', 'mini', 'headlight'] if c in set(d.camera)]
    pairs = sc[sc.hours >= args.min_hours_pair].pivot(index='subject_id', columns='camera', values='pitch_mean').dropna(thresh=2)
    pair_subjects = sorted(pairs.index); print('within-child pairs:', len(pair_subjects))
    json.dump({'pair_subjects': pair_subjects, 'cameras': cams}, open(f'{args.out}/meta.json', 'w'))

    # F1 distributions
    fig, ax = plt.subplots(figsize=(4.2, 3.4)); style(ax)
    data = [d[d.camera == c].pitch_down_deg.dropna() for c in cams]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.2))
    ax.set_xticks(range(1, len(cams) + 1)); ax.set_xticklabels(cams)
    for patch, c in zip(bp['boxes'], cams): patch.set(facecolor=CAM_COLORS[c], alpha=0.35, edgecolor=CAM_COLORS[c])
    ax.set_ylabel('per-video camera pitch (deg below horizontal)'); ax.set_title('IMU camera pitch by camera version (all videos)', fontsize=9)
    fig.savefig(f'{args.out}/imu_fig1_pitch_by_camera.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    # F2 paired within child
    if pair_subjects and 'bones' in cams and 'mini' in cams:
        fig, ax = plt.subplots(figsize=(3.6, 3.8)); style(ax)
        for s in pair_subjects:
            r = sc[sc.subject_id == s].set_index('camera')
            if 'bones' in r.index and 'mini' in r.index:
                ax.plot([0, 1], [r.loc['bones', 'pitch_mean'], r.loc['mini', 'pitch_mean']], color=GRID, lw=1, zorder=1)
                for xi, c in enumerate(['bones', 'mini']): ax.scatter(xi, r.loc[c, 'pitch_mean'], s=28, color=CAM_COLORS[c], edgecolor=SURF, linewidth=0.8, zorder=2)
        mean = sc[sc.subject_id.isin(pair_subjects)].groupby('camera').pitch_mean.mean()
        for xi, c in enumerate(['bones', 'mini']):
            ax.hlines(mean[c], xi - 0.18, xi + 0.18, color=CAM_COLORS[c], lw=2); ax.text(xi + 0.2, mean[c], f'{mean[c]:.1f}°', fontsize=8, color=INK2, va='center')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['bones', 'mini']); ax.set_xlim(-0.5, 1.6); ax.set_ylabel('mean per-video pitch (deg below horizontal)')
        ax.set_title(f'Within-child camera pitch at the switch\n({len(pair_subjects)} children, ≥{args.min_hours_pair:g} h each; bar = mean; Δ = {mean["mini"] - mean["bones"]:+.1f}°)', fontsize=9)
        fig.savefig(f'{args.out}/imu_fig2_paired_by_child.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    # F3 time course
    if pair_subjects:
        n = len(pair_subjects); ncol = min(n, 6); nrow = int(np.ceil(n / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow), squeeze=False, sharey=True)
        for i, s in enumerate(pair_subjects):
            ax = axs[divmod(i, ncol)]; style(ax); v = d[d.subject_id == s]
            for c in cams:
                w = v[v.camera == c]; ax.scatter(w.date, w.pitch_down_deg, s=8 + 10 * (w.duration_sec / 3600).clip(0, 2), color=CAM_COLORS[c], alpha=0.6, edgecolor='none', label=c)
            ax.set_title(s, fontsize=8); ax.tick_params(axis='x', labelrotation=45, labelsize=6)
        for k in range(n, nrow * ncol): axs[divmod(k, ncol)].axis('off')
        axs[0, 0].set_ylabel('pitch (deg below horizontal)', fontsize=8); axs[0, 0].legend(fontsize=7, frameon=False, loc='upper left')
        fig.subplots_adjust(hspace=0.75, wspace=0.15)
        fig.suptitle('Per-video IMU camera pitch over time (dot size ~ duration; + = pointing down)', fontsize=10, y=0.995)
        fig.savefig(f'{args.out}/imu_fig3_timecourse_by_child.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    # F4 pitch vs pose hand position (per video)
    if args.pose_video_summary:
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
        for ax, (m, lab) in zip(axs, [('ego_y_mean', 'mean ego-hand y (1 = bottom)'), ('p_ego_hand', 'P(ego hand in frame)')]):
            style(ax)
            for c in cams:
                s = d[(d.camera == c) & d[m].notna() & (d.n_frames >= 300)]
                ax.scatter(s.pitch_down_deg, s[m], s=6, color=CAM_COLORS[c], alpha=0.35, edgecolor='none', label=f'{c} (r={s.pitch_down_deg.corr(s[m]):+.2f})')
            ax.set_xlabel('IMU pitch (deg below horizontal)'); ax.set_ylabel(lab); ax.legend(fontsize=7, frameon=False)
        fig.suptitle('Direct camera pitch (IMU) vs the pose-based hand indicators, per video', fontsize=10, y=1.02)
        fig.savefig(f'{args.out}/imu_fig4_pitch_vs_pose.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('ANALYZE_DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract'); e.add_argument('--pull_root', required=True); e.add_argument('--out', required=True); e.add_argument('--procs', type=int, default=12)
    a = sub.add_parser('analyze'); a.add_argument('--imu', required=True); a.add_argument('--crosswalk', required=True); a.add_argument('--out', required=True)
    a.add_argument('--pose_video_summary'); a.add_argument('--exclude_cameras', default='headlight'); a.add_argument('--min_hours_pair', type=float, default=2.0)
    a.add_argument('--fov', action='append', default=[], help='camera=vertical_fov_deg (portrait), e.g. bones=110 mini=130')
    a.add_argument('--release_ids', help='file of video_ids that constitute the release; rows outside it are dropped')
    args = ap.parse_args(); extract(args) if args.cmd == 'extract' else analyze(args)
