"""Re-evaluate SAVED checkpoints under the corrected 4AFC (OOV categories score chance).

The eval fix changes evaluation only, so ch8/ch9 do not need retraining — their models and
features are intact. This reloads each checkpoint and rescores it over the FULL category set,
reporting the corrected accuracy next to what was published.

usage: python src/reeval_saved.py --runs 'runs_oak/vlm_enc/runs/E_*' --eval-cache-map auto
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd, torch

from train import build_vocab, encode          # noqa
from train_region_mil import RegionMIL, eval_4afc_region, load_region_cache
from common import frame_key                    # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="glob of run dirs holding model.pt + vocab.json")
ap.add_argument("--eval-frames", default="manifests/eval_frames_konkle.parquet")
ap.add_argument("--eval-root", default="emb_enc_eval", help="R=1 (whole-frame) eval caches")
ap.add_argument("--grid-eval-root", default="emb_enc_grid_eval", help="R=16 (grid) eval caches")
ap.add_argument("--out", default="results/reeval_corrected.csv")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ev = pd.read_parquet(a.eval_frames)
print(f"eval set: {len(ev)} images, {ev.category.nunique()} categories")

# run name -> encoder tag, e.g. E_dinov3b_ots_grid_s0 -> dinov3b_ots, readout 'grid'
def parse(run):
    m = re.match(r"E_(.+?)_(wf|wfc|grid)_s(\d+)$", run)
    if m: return m.group(1), m.group(2), int(m.group(3))
    m = re.match(r"(max|soft\d+|attn|cap)_s(\d+)$", run)
    if m: return m.group(1), "arch", int(m.group(2))
    return None, None, None

cache = {}
def get_eval(tag, readout):
    root = a.grid_eval_root if readout == "grid" else a.eval_root
    key = (root, tag)
    if key not in cache:
        d = f"{root}/{tag}_konkle"
        if not os.path.exists(f"{d}/index.parquet"):
            return None
        cache[key] = load_region_cache(d)
    return cache[key]

rows = []
for rd in sorted(glob.glob(a.runs)):
    run = os.path.basename(rd)
    tag, readout, seed = parse(run)
    if tag is None or readout == "arch":
        continue                       # arch models need train_arch's class; handled separately
    ec = get_eval(tag, readout)
    if ec is None:
        print(f"  skip {run}: no eval cache for {tag}/{readout}"); continue
    emb, lut = ec
    try:
        vocab = json.load(open(f"{rd}/vocab.json"))
        sd = torch.load(f"{rd}/model.pt", map_location=dev, weights_only=True)
    except Exception as e:
        print(f"  skip {run}: {e}"); continue
    # RegionMIL: vproj = Sequential(LayerNorm(emb_dim), Dropout, Linear(emb_dim, dim))
    D = int(sd["vproj.2.weight"].shape[1]) if "vproj.2.weight" in sd else None
    if D is None:
        print(f"  skip {run}: cannot infer emb_dim from checkpoint"); continue
    m = RegionMIL(len(vocab), emb_dim=D).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    acc, det = eval_4afc_region(m, emb, lut, ev, vocab, dev, return_detail=True)
    rows.append(dict(run=run, family=re.sub(r"_s\d+$", "", run), encoder=tag, readout=readout,
                     seed=seed, corrected=round(100*acc, 2),
                     n_scored=det["n_scored"], n_oov=det["n_oov"], n_total=det["n_total"]))
    print(f"  {run:32s} corrected {100*acc:5.2f}  ({det['n_scored']} scored + {det['n_oov']} OOV)")

df = pd.DataFrame(rows)
if len(df):
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)
    g = df.groupby("family").agg(n=("seed","size"), corrected=("corrected","mean"),
                                 sd=("corrected","std"), oov=("n_oov","mean")).round(2)
    print("\n=== corrected (OOV at chance, full category set) ===")
    print(g.sort_values("corrected", ascending=False).to_string())
    print(f"\nwrote {a.out}")
