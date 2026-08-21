#!/usr/bin/env python3
"""Hand availability / position by child x camera version (bones = V2 GoPro Bones, mini = V3 GoPro HERO).

Why: the V3 mount points ~20 deg further up than V2 (partly offset by a wider FOV), and for some
children the camera appears tilted up. The child's own hands sit at the bottom of the frame, so
their prevalence and vertical position are both a scientific quantity and a camera-angle indicator.

Operationalization (from the whole-body pose CSV, one row per detected person):
  * ego hand  := a hand (left/right_hand_in_image == 1) on a detection with NO face (face_in_image == 0).
               In 2025.2 these concentrate in the bottom 20-30% of the frame (the wearer's own hands).
  * caregiver hand := hand on a face-present detection (control: everything shifts down if the camera tilts up).
  * positions are hand-bbox centers normalized by the video's frame size (y = 0 top, 1 bottom).
  * optional angular conversion: theta_below_axis = (y - 0.5) * vFOV (equidistant approx), vFOV per camera.

Inputs: pose CSV (29-col schema), Airtable crosswalk TSV (video_id, subject_id, camera, date, ...),
frame-dims TSV(s) (video_id, w, h, n_frames). Outputs: per-video + per-subject-x-camera tables,
heatmap arrays, and figures. Aggregates only -- no frames leave the node.

usage (node):
  python pose_hand_camera.py --csv <pose_1fps_bbox_limbs.csv> --crosswalk <tsv> --dims <tsv> [--dims <tsv>] \
      --out <dir> [--min_frames 60] [--exclude_cameras headlight] [--fov bones=110 --fov mini=130]
"""
import argparse, os, re, json
import numpy as np, pandas as pd

CAM_COLORS = {'bones': '#2a78d6', 'mini': '#eb6834', 'headlight': '#1baf7a'}   # categorical slots 1-3
SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
            '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
DIV = ['#2a78d6', '#f0efec', '#e34948']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e8e7e3', '#fcfcfb'
YB, XB = 20, 12    # heatmap bins (rows = y top->bottom, cols = x)
BBOX_RE = r'\[\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]'


def parse_bbox(s):
    b = s.astype(str).str.extract(BBOX_RE).astype(float); b.columns = ['x1', 'y1', 'x2', 'y2']; return b


def load(args):
    use = ['superseded_gcp_name_feb25', 'time_in_extended_iso', 'person_detected', 'face_in_image', 'body_in_image',
           'left_hand_in_image', 'right_hand_in_image', 'left_hand_bounding_box_xyxy', 'right_hand_bounding_box_xyxy',
           'face_bounding_box_xyxy', 'left_hand_score', 'right_hand_score']
    df = pd.read_csv(args.csv, usecols=use, low_memory=False).rename(columns={'superseded_gcp_name_feb25': 'video_id', 'time_in_extended_iso': 't'})
    print('rows', len(df), 'videos', df.video_id.nunique(), flush=True)
    dims = pd.concat([pd.read_csv(p, sep='\t', header=None, names=['video_id', 'w', 'h', 'n_jpg']) for p in args.dims]).drop_duplicates('video_id')
    xw = pd.read_csv(args.crosswalk, sep='\t', low_memory=False)
    xw = xw.rename(columns={c: c.strip() for c in xw.columns})
    # create_csv.py strips '_processed' from dir names (the 55 '_blackout_processed' videos) -> match that convention
    dims['video_id'] = dims.video_id.str.replace('_processed', '', regex=False)
    xw['video_id'] = xw.video_id.str.replace('_processed', '', regex=False)
    xw['date'] = pd.to_datetime(xw['date'], errors='coerce')
    for c in ['age (years)', 'duration_sec']:          # Airtable exports carry '#ERROR!' strings in numeric columns
        xw[c] = pd.to_numeric(xw[c], errors='coerce')
    return df, dims, xw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--crosswalk', required=True)
    ap.add_argument('--dims', action='append', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--min_frames', type=int, default=60)
    ap.add_argument('--exclude_cameras', default='headlight')
    ap.add_argument('--fov', action='append', default=[], help='camera=vertical_fov_deg, e.g. bones=110')
    ap.add_argument('--min_hours_pair', type=float, default=2.0, help='min hours per camera for a child to count as a within-child pair')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fov = {k: float(v) for k, v in (s.split('=') for s in args.fov)}
    excl = set(filter(None, args.exclude_cameras.split(',')))

    df, dims, xw = load(args)
    # ---- frames per video (rows are persons; frames with no person appear as one row each)
    nfr = df.groupby('video_id').t.nunique().rename('n_frames')
    meta = nfr.to_frame().join(dims.set_index('video_id'), how='left').join(
        xw.set_index('video_id')[['subject_id', 'camera', 'dataset', 'date', 'age (years)', 'duration_sec']], how='left')
    meta = meta.rename(columns={'age (years)': 'age_years'})
    bad = meta.w.isna() | (meta.w == 0); print('videos w/o frame dims:', int(bad.sum()))
    meta = meta[~bad & meta.camera.notna() & ~meta.camera.isin(excl) & (meta.n_frames >= args.min_frames)]
    print('videos kept', len(meta), meta.camera.value_counts().to_dict(), flush=True)
    df = df[df.video_id.isin(meta.index)]

    # ---- long table of hands
    det = df[df.person_detected == 1]
    hands = []
    for side in ['left', 'right']:
        m = det[f'{side}_hand_in_image'] == 1
        b = parse_bbox(det.loc[m, f'{side}_hand_bounding_box_xyxy'])
        h = pd.DataFrame({'video_id': det.loc[m, 'video_id'].values, 't': det.loc[m, 't'].values, 'side': side,
                          'cx': ((b.x1 + b.x2) / 2).values, 'cy': ((b.y1 + b.y2) / 2).values,
                          'ego': (det.loc[m, 'face_in_image'] == 0).values, 'score': det.loc[m, f'{side}_hand_score'].values})
        hands.append(h)
    hands = pd.concat(hands, ignore_index=True)
    hands = hands.join(meta[['w', 'h', 'camera', 'subject_id']], on='video_id')
    hands['x'] = (hands.cx / hands.w).clip(0, 1); hands['y'] = (hands.cy / hands.h).clip(0, 1)
    hands['theta'] = [(y - 0.5) * fov.get(c, np.nan) for y, c in zip(hands.y, hands.camera)]
    # faces (caregiver control)
    fm = det.face_in_image == 1
    fb = parse_bbox(det.loc[fm, 'face_bounding_box_xyxy'])
    faces = pd.DataFrame({'video_id': det.loc[fm, 'video_id'].values, 't': det.loc[fm, 't'].values,
                          'cy': ((fb.y1 + fb.y2) / 2).values}).join(meta[['h']], on='video_id')
    faces['y'] = (faces.cy / faces.h).clip(0, 1)
    print('hands', len(hands), 'ego', int(hands.ego.sum()), 'faces', len(faces), flush=True)

    # ---- per-frame flags -> per-video summary
    ego = hands[hands.ego]; care = hands[~hands.ego]
    vs = meta.copy()
    vs['ego_hand_frames'] = ego.groupby('video_id').t.nunique().reindex(vs.index).fillna(0).astype(int)
    vs['any_hand_frames'] = hands.groupby('video_id').t.nunique().reindex(vs.index).fillna(0).astype(int)
    vs['face_frames'] = faces.groupby('video_id').t.nunique().reindex(vs.index).fillna(0).astype(int)
    vs['p_ego_hand'] = vs.ego_hand_frames / vs.n_frames
    vs['p_any_hand'] = vs.any_hand_frames / vs.n_frames
    vs['p_face'] = vs.face_frames / vs.n_frames
    g = ego.groupby('video_id')
    vs['ego_y_mean'] = g.y.mean(); vs['ego_y_median'] = g.y.median(); vs['ego_x_mean'] = g.x.mean()
    vs['p_ego_y_gt_0p8'] = g.y.apply(lambda s: (s > 0.8).mean()); vs['n_ego_hands'] = g.size()
    vs['ego_theta_mean'] = g.theta.mean()
    vs['care_y_mean'] = care.groupby('video_id').y.mean()
    vs['face_y_mean'] = faces.groupby('video_id').y.mean()
    vs['hours'] = vs.n_frames / 3600; vs['aspect'] = (vs.w / vs.h).round(3)
    vs.reset_index().to_csv(f'{args.out}/video_summary.csv', index=False)

    # ---- subject x camera (frame-weighted prevalence; hand-weighted positions)
    def agg(gdf):
        e = ego[ego.video_id.isin(gdf.index)]; f = faces[faces.video_id.isin(gdf.index)]; c = care[care.video_id.isin(gdf.index)]
        return pd.Series({'n_videos': len(gdf), 'hours': gdf.hours.sum(), 'n_frames': gdf.n_frames.sum(),
                          'p_ego_hand': gdf.ego_hand_frames.sum() / gdf.n_frames.sum(),
                          'p_any_hand': gdf.any_hand_frames.sum() / gdf.n_frames.sum(),
                          'p_face': gdf.face_frames.sum() / gdf.n_frames.sum(),
                          'ego_y_mean': e.y.mean(), 'ego_y_median': e.y.median(), 'p_ego_y_gt_0p8': (e.y > 0.8).mean(),
                          'ego_x_mean': e.x.mean(), 'ego_theta_mean': e.theta.mean(), 'n_ego_hands': len(e),
                          'care_y_mean': c.y.mean(), 'face_y_mean': f.y.mean(),
                          'first_date': gdf.date.min(), 'last_date': gdf.date.max(), 'age_mean': gdf.age_years.mean(),
                          'frac_4x3': (gdf.aspect > 0.7).mean()})
    sc = vs.groupby(['subject_id', 'camera']).apply(agg).reset_index()
    sc.to_csv(f'{args.out}/subject_camera_summary.csv', index=False)
    pooled = vs.groupby('camera').apply(agg).reset_index(); pooled.to_csv(f'{args.out}/camera_summary.csv', index=False)
    print(pooled[['camera', 'n_videos', 'hours', 'p_ego_hand', 'ego_y_mean', 'p_ego_y_gt_0p8', 'face_y_mean']].round(3).to_string(), flush=True)

    # ---- heatmaps: ego-hand density per subject x camera and pooled per camera
    def H(e):
        h, _, _ = np.histogram2d(e.y, e.x, bins=[YB, XB], range=[[0, 1], [0, 1]]); return h
    hm = {f'pooled|{c}': H(ego[ego.camera == c]) for c in vs.camera.unique()}
    for (s, c), gdf in vs.groupby(['subject_id', 'camera']):
        hm[f'{s}|{c}'] = H(ego[ego.video_id.isin(gdf.index)])
    np.savez_compressed(f'{args.out}/ego_hand_heatmaps.npz', **hm)

    # ---- within-child pairs: children with >= min_hours_pair on two cameras
    cams = [c for c in ['bones', 'mini', 'headlight'] if c in set(vs.camera)]
    pairs = sc[sc.hours >= args.min_hours_pair].pivot(index='subject_id', columns='camera', values='hours').dropna(thresh=2)
    pair_subjects = sorted(pairs.index)
    print('within-child pairs:', len(pair_subjects), pair_subjects, flush=True)
    json.dump({'pair_subjects': pair_subjects, 'cameras': cams, 'fov': fov}, open(f'{args.out}/meta.json', 'w'))
    make_figures(args, vs, sc, hm, pair_subjects, cams, fov)


def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=2, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.set_facecolor(SURF)


def make_figures(args, vs, sc, hm, pair_subjects, cams, fov):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list('seq_blue', ['#ffffff'] + SEQ_BLUE)
    div = LinearSegmentedColormap.from_list('div', DIV)
    plt.rcParams.update({'font.family': 'sans-serif', 'text.color': INK, 'axes.labelcolor': INK2, 'figure.facecolor': SURF, 'savefig.facecolor': SURF})
    two = [c for c in cams if c in ('bones', 'mini')]

    # F1: pooled heatmaps + difference
    def dens(h): return 100 * h / max(h.sum(), 1)
    fig, axs = plt.subplots(1, len(two) + (1 if len(two) == 2 else 0), figsize=(3.2 * (len(two) + 1), 5.2))
    axs = np.atleast_1d(axs)
    vmax = max(dens(hm[f'pooled|{c}']).max() for c in two)
    for ax, c in zip(axs, two):
        im = ax.imshow(dens(hm[f'pooled|{c}']), cmap=seq, vmin=0, vmax=vmax, extent=[0, 1, 1, 0], aspect=1 / (YB / XB) * (YB / XB) * 1.0)
        r = sc[sc.camera == c]; ax.set_title(f'{c}: ego-hand density\n{vs[vs.camera == c].hours.sum():.0f} h, {int(vs[vs.camera == c].n_ego_hands.sum())} hands', fontsize=9, color=INK)
        ax.set_xlabel('x (normalized)'); ax.set_ylabel('y (0 = top, 1 = bottom)'); ax.tick_params(labelsize=7, colors=INK2)
    if len(two) == 2:
        d = dens(hm['pooled|mini']) - dens(hm['pooled|bones']); lim = np.abs(d).max()
        im2 = axs[-1].imshow(d, cmap=div, vmin=-lim, vmax=lim, extent=[0, 1, 1, 0])
        axs[-1].set_title('mini − bones (pct-point density)', fontsize=9, color=INK); axs[-1].tick_params(labelsize=7, colors=INK2)
        fig.colorbar(im2, ax=axs[-1], fraction=0.05, pad=0.04).ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=list(axs[:len(two)]), fraction=0.03, pad=0.03, label='% of ego hands per cell').ax.tick_params(labelsize=7)
    fig.suptitle('Where the child\'s own hands appear in the frame, by camera version (pooled over children)', fontsize=10, color=INK, y=1.02)
    fig.savefig(f'{args.out}/fig1_heatmap_pooled.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # F2: per-child heatmaps, transition children only (rows = children, cols = cameras)
    if pair_subjects and len(two) == 2:
        n = len(pair_subjects); ncol = 2 * min(n, 6); nrow = int(np.ceil(n / 6))
        fig, axs = plt.subplots(nrow, ncol, figsize=(1.35 * ncol, 2.6 * nrow), squeeze=False)
        for i, s in enumerate(pair_subjects):
            r, c0 = divmod(i, 6)
            for j, c in enumerate(two):
                ax = axs[r, 2 * c0 + j]; key = f'{s}|{c}'
                if key in hm and hm[key].sum() > 0:
                    ax.imshow(dens(hm[key]), cmap=seq, vmin=0, vmax=vmax * 1.5, extent=[0, 1, 1, 0])
                row = sc[(sc.subject_id == s) & (sc.camera == c)]
                hrs = float(row.hours.iloc[0]) if len(row) else 0
                ax.set_title(f'{s[-5:]} {c}\n{hrs:.0f} h', fontsize=7, color=CAM_COLORS[c]); ax.set_xticks([]); ax.set_yticks([])
        for k in range(n, nrow * 6):
            r, c0 = divmod(k, 6); axs[r, 2 * c0].axis('off'); axs[r, 2 * c0 + 1].axis('off')
        fig.suptitle('Ego-hand density per child: bones (V2) vs mini (V3)', fontsize=10, color=INK, y=1.02)
        fig.savefig(f'{args.out}/fig2_heatmap_by_child.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # F3: paired within-child comparison (prevalence, vertical position, face position, angle)
    if pair_subjects and len(two) == 2:
        metrics = [('p_ego_hand', 'P(frame has an ego hand)'), ('ego_y_mean', 'mean ego-hand y (1 = bottom)'),
                   ('face_y_mean', 'mean face y (caregiver control)')] + ([('ego_theta_mean', 'mean ego-hand angle below axis (deg)')] if fov else [])
        fig, axs = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.8))
        for ax, (m, lab) in zip(axs, metrics):
            style(ax)
            for s in pair_subjects:
                r = sc[sc.subject_id == s].set_index('camera')
                if all(c in r.index for c in two):
                    ax.plot([0, 1], [r.loc['bones', m], r.loc['mini', m]], color=GRID, lw=1, zorder=1)
                    for xi, c in enumerate(two):
                        ax.scatter(xi, r.loc[c, m], s=28, color=CAM_COLORS[c], edgecolor=SURF, linewidth=0.8, zorder=2)
            mean = sc[sc.subject_id.isin(pair_subjects)].groupby('camera')[m].mean()
            for xi, c in enumerate(two):
                ax.hlines(mean[c], xi - 0.18, xi + 0.18, color=CAM_COLORS[c], lw=2)
                ax.text(xi + 0.2, mean[c], f'{mean[c]:.2f}', fontsize=8, color=INK2, va='center')
            ax.set_xticks([0, 1]); ax.set_xticklabels(two); ax.set_xlim(-0.5, 1.6); ax.set_title(lab, fontsize=9, color=INK)
        axs[0].legend(handles=[plt.Line2D([], [], marker='o', ls='', color=CAM_COLORS[c], label=c) for c in two], fontsize=8, frameon=False, loc='upper left')
        fig.suptitle(f'Within-child change at the camera switch ({len(pair_subjects)} children, ≥{args.min_hours_pair:g} h per camera); bar = mean', fontsize=10, color=INK, y=1.02)
        fig.savefig(f'{args.out}/fig3_paired_by_child.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # F4: per-video time course for transition children (small multiples)
    if pair_subjects:
        n = len(pair_subjects); ncol = min(n, 6); nrow = int(np.ceil(n / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow), squeeze=False, sharey=True)
        for i, s in enumerate(pair_subjects):
            ax = axs[divmod(i, ncol)]; style(ax); v = vs[vs.subject_id == s]
            for c in cams:
                w = v[v.camera == c]
                ax.scatter(w.date, w.p_ego_hand, s=8 + 20 * w.hours.clip(0, 2), color=CAM_COLORS[c], alpha=0.6, edgecolor='none', label=c)
            ax.set_title(s, fontsize=8, color=INK); ax.tick_params(axis='x', labelrotation=45, labelsize=6)
        for k in range(n, nrow * ncol): axs[divmod(k, ncol)].axis('off')
        axs[0, 0].set_ylabel('P(ego hand in frame)', fontsize=8)
        axs[0, 0].legend(fontsize=7, frameon=False, loc='upper left')
        fig.suptitle('Per-video ego-hand prevalence over time (dot size ~ hours)', fontsize=10, color=INK, y=1.02)
        fig.savefig(f'{args.out}/fig4_timecourse_by_child.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # F5: distribution of per-video metrics by camera (all children), as boxplots
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, (m, lab) in zip(axs, [('p_ego_hand', 'P(ego hand in frame)'), ('ego_y_mean', 'mean ego-hand y'), ('face_y_mean', 'mean face y')]):
        style(ax); data = [vs[vs.camera == c][m].dropna() for c in cams]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.2))
        ax.set_xticks(range(1, len(cams) + 1)); ax.set_xticklabels(cams)
        for patch, c in zip(bp['boxes'], cams): patch.set(facecolor=CAM_COLORS[c], alpha=0.35, edgecolor=CAM_COLORS[c])
        ax.set_title(lab, fontsize=9, color=INK)
    fig.suptitle('Per-video distributions by camera (all children, videos ≥ %d frames)' % args.min_frames, fontsize=10, color=INK, y=1.02)
    fig.savefig(f'{args.out}/fig5_video_distributions.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('figures written to', args.out)


if __name__ == '__main__':
    main()
