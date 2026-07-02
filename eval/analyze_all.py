import json
from pathlib import Path
from statistics import mean, pstdev

R = Path("runs")

def load(name):
    p = R / name / "log.json"
    return json.loads(p.read_text()) if p.exists() else None

def best(name):
    lg = load(name)
    return max(r["acc_4afc"] for r in lg) if lg else None

def final(name):
    lg = load(name)
    return lg[-1] if lg else None

def ms(vals):
    a = [v for v in vals if v is not None]
    return mean(a), (pstdev(a) if len(a) > 1 else 0.0)

CHILDREN = ["S00360001", "S00240001", "S00370002"]
SEEDS = [0, 1, 2]

print("="*64)
print("EXP B — robustness: aligned vs random (best 4AFC, mean±sd over 3 seeds)")
print("="*64)
print(f"{'held-out child':16} {'aligned':>14} {'random':>14} {'gap':>7}")
gaps = []
for ch in CHILDREN:
    al = [best(f"B_aligned_{ch}_s{s}") for s in SEEDS]
    rn = [best(f"B_random_{ch}_s{s}") for s in SEEDS]
    am, asd = ms(al); rm, rsd = ms(rn)
    gaps.append(am - rm)
    print(f"{ch:16} {am*100:6.1f} ± {asd*100:3.1f}   {rm*100:6.1f} ± {rsd*100:3.1f}   {(am-rm)*100:+5.1f}")
print(f"{'MEAN GAP':16} {'':>14} {'':>14} {mean(gaps)*100:+5.1f}")

print("\n" + "="*64)
print("EXP A — unfiltered divergence arm (held-out S00360001)")
print("="*64)
a = load("A_unfiltered_S00360001")
if a:
    vmin = min(r["val_loss"] for r in a); vlast = a[-1]["val_loss"]
    print(f"  1.14M pairs. best 4AFC {max(r['acc_4afc'] for r in a)*100:.1f}  final {a[-1]['acc_4afc']*100:.1f}")
    print(f"  val_loss: start {a[0]['val_loss']:.3f}  min {vmin:.3f}  final {vlast:.3f}  "
          f"({'RISES (diverges)' if vlast>vmin+0.02 else 'stable'})")
    print(f"  train_loss: start {a[0]['train_loss']:.3f}  final {a[-1]['train_loss']:.3f}")
# compare with aligned/random at same child (seed 0)
print("  vs (best 4AFC, S00360001, seed0):")
print(f"    aligned(137k)={best('B_aligned_S00360001_s0')*100:.1f}  "
      f"random(137k)={best('B_random_S00360001_s0')*100:.1f}  "
      f"unfiltered(1.14M)={max(r['acc_4afc'] for r in a)*100:.1f}")

print("\n" + "="*64)
print("EXP C — threshold sweep (held-out S00360001, mean±sd over seeds)")
print("="*64)
sizes = {"0.24": 137179, "0.26": 21877, "0.28": 3799}
print(f"{'min-clip':10} {'pairs':>8} {'best 4AFC':>16}")
# 0.24 == B_aligned_S00360001
al24 = [best(f"B_aligned_S00360001_s{s}") for s in SEEDS]
m, sd = ms(al24); print(f"{'0.24':10} {sizes['0.24']:>8} {m*100:8.1f} ± {sd*100:.1f}")
for thr in ["0.26", "0.28"]:
    vals = [best(f"C_thr{thr}_s{s}") for s in SEEDS]
    m, sd = ms(vals); print(f"{thr:10} {sizes[thr]:>8} {m*100:8.1f} ± {sd*100:.1f}")

print("\n" + "="*64)
print("EXP D — region grounding 2x2 (best 4AFC; 50k detection-video pairs)")
print("="*64)
print(f"{'':14} {'frame-eval':>12} {'crop-eval':>12}")
for tr in ["frame", "crop"]:
    row = []
    for ev in ["frameeval", "cropeval"]:
        b = best(f"D_{tr}_{ev}")
        row.append(f"{b*100:.1f}" if b else "  -")
    print(f"{tr+'-train':14} {row[0]:>12} {row[1]:>12}")
