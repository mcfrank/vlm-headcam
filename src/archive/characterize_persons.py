"""Before building a child-vs-adult hand classifier, ask the data: are the two populations
actually separable in the pose output? Hypothesis (Mike): the child's own hands enter from
the BOTTOM of the frame and are UNATTACHED to a body/face (the wearer's torso/face aren't in
their own headcam); adults appear as full upper bodies with a visible face. Characterize each
detected person by which parts are visible and where they sit."""
import os, sys, random
import numpy as np
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_lib import load_pose, POSE_ROOT, NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LHIP, RHIP, LWRI, RWRI, LHAND, RHAND
FH, FW = 910.0, 512.0
TH = 0.3

def feats(P, b):
    kp, sc = P["kp"], P["score"]; vis = sc > TH
    face = float(np.mean(sc[[NOSE, LEYE, REYE, LEAR, REAR]]))
    torso = float(np.mean(sc[[LSHO, RSHO, LHIP, RHIP]]))
    hand = float(max(np.mean(sc[LHAND]), np.mean(sc[RHAND])))
    wr_y = [kp[i, 1] for i in (LWRI, RWRI) if vis[i]]
    wristy = float(np.mean(wr_y) / FH) if wr_y else np.nan
    vy = kp[vis, 1]
    cy = float(np.mean(vy) / FH) if len(vy) else np.nan
    return dict(face=face, torso=torso, hand=hand, wristy=wristy, cy=cy,
                bbox_h=float((b[3] - b[1]) / FH), bbox_bottom=float(b[3] / FH), bbox_top=float(b[1] / FH),
                nkp_vis=int(vis.sum()))

rng = random.Random(0)
dirs = [d for d in os.listdir(POSE_ROOT) if d.endswith("_processed")]
rng.shuffle(dirs)
rows = []
for d in dirs[:400]:
    pk = [f for f in os.listdir(f"{POSE_ROOT}/{d}") if f.endswith(".pkl")]
    for pf in rng.sample(pk, min(8, len(pk))):
        pose = load_pose(f"{POSE_ROOT}/{d}/{pf}")
        for P in pose["persons"]:
            i = P["id"]
            if isinstance(i, int) and i < len(pose["bboxes"]) and P["kp"].shape[0] == 133:
                rows.append(feats(P, pose["bboxes"][i]))
    if len(rows) > 12000:
        break

import pandas as pd
df = pd.DataFrame(rows)
print(f"persons: {len(df)}")
print("\n-- visibility of parts (mean score) --")
print(f"  face visible (>0.3): {(df.face>0.3).mean():.1%}   torso: {(df.torso>0.3).mean():.1%}   hand: {(df.hand>0.3).mean():.1%}")
# candidate populations
adult = df[(df.face > 0.3) & (df.torso > 0.3)]           # full upper body + face
orphan = df[(df.hand > 0.3) & (df.face < 0.15) & (df.torso < 0.2)]  # hands, no face/torso
print(f"\n-- ADULT-like (face+torso visible): {len(adult)} ({len(adult)/len(df):.1%}) --")
print(f"   centroid_y median={adult.cy.median():.2f}  bbox_top median={adult.bbox_top.median():.2f}  bbox_h median={adult.bbox_h.median():.2f}")
print(f"-- ORPHAN-hand (hand, no face/torso): {len(orphan)} ({len(orphan)/len(df):.1%}) --")
print(f"   wrist_y median={orphan.wristy.median():.2f}  centroid_y median={orphan.cy.median():.2f}  bbox_bottom median={orphan.bbox_bottom.median():.2f}  bbox_top median={orphan.bbox_top.median():.2f}")
# where are visible hands overall, split by whether the same person has a face
hw = df[df.hand > 0.3]
print(f"\n-- all persons with a visible hand: {len(hw)} --")
print(f"   with face visible: wrist_y median={hw[hw.face>0.3].wristy.median():.2f} (n={ (hw.face>0.3).sum()})")
print(f"   NO face visible:   wrist_y median={hw[hw.face<0.15].wristy.median():.2f} (n={ (hw.face<0.15).sum()})")
print("\ninterpretation: if orphan-hand persons cluster LOW (wrist_y>0.6, bbox touches bottom)")
print("and adults sit HIGHER with faces, the child/adult split is recoverable by a simple rule.")
