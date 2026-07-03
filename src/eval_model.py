"""Eval a saved region-MIL model on any eval_frames (no retraining). Also rebuild a within-child
eval restricted to a reference category set (for a clean within- vs cross-child comparison)."""
import sys, json, os
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from train_region_mil import RegionMIL, load_region_cache, eval_4afc_region
W = "/data2/mcfrank/vlm-headcam"; dev = "cuda" if torch.cuda.is_available() else "cpu"

def do_eval(run, eval_frames, cache=f"{W}/emb_reg"):
    vocab = json.load(open(f"{W}/runs/{run}/vocab.json"))
    emb, lut = load_region_cache(cache)
    ev = pd.read_parquet(eval_frames)
    m = RegionMIL(len(vocab), 512).to(dev)
    m.load_state_dict(torch.load(f"{W}/runs/{run}/model.pt", map_location=dev)); m.eval()
    acc = eval_4afc_region(m, emb, lut, ev, vocab, dev, seed=0)
    # how many eval categories are actually in this model's vocab (scored, not skipped)
    from train import encode
    scored = sum(1 for c in ev.category.unique() if encode(c, vocab, 16))
    print(f"  {run:18s} on {os.path.basename(eval_frames):26s} 4AFC={acc*100:.1f}  (cats scored {scored}/{ev.category.nunique()})")
    return acc

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rebuild_matched":
        # restrict a within-child eval to the reference (cross-child) category set
        CH = sys.argv[2]
        ref = set(pd.read_parquet(f"{W}/manifests/eval_frames.parquet").category.unique())
        ev = pd.read_parquet(f"{W}/manifests/eval_frames_{CH}.parquet")
        evm = ev[ev.category.isin(ref)]
        evm.to_parquet(f"{W}/manifests/eval_frames_{CH}_matched.parquet")
        print(f"matched eval: {evm.category.nunique()} categories (of {len(ref)} ref), {len(evm)} frames")
    else:
        do_eval(*sys.argv[1:4])
