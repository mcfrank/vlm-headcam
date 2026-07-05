"""Aggregate the grid seed error bars (test-60) and eval the whole-frame grid on dev-117."""
import json
import numpy as np
import pandas as pd
import torch
from train import TwoTower, eval_4afc
from common import load_emb_cache

# test-60, seeds 0/1/2
data = {
    "no-MIL base":    [53.1, 53.4, 49.9], "no-MIL T1>=50": [62.5, 62.0, 60.2], "no-MIL T2": [74.5, 75.8, 74.7],
    "MIL base":       [63.7, 63.5, 60.7], "MIL T1>=50":    [67.2, 69.3, 68.0], "MIL T2":    [84.0, 78.9, 81.0],
}
print("=== test-60 (mean +/- sd over 3 seeds) ===")
for k, v in data.items():
    print(f"  {k:16s} {np.mean(v):5.1f} +/- {np.std(v, ddof=1):.1f}")

print("=== whole-frame grid on dev-117 (seed 0) ===")
emb, lut = load_emb_cache("emb_konkle_dev_cls")
ev = pd.read_parquet("manifests/eval_frames_konkle_dev.parquet")
dv = "cuda"
for run in ["G_base_wf_s0", "G_t1_ge50_wf_s0", "G_t1_ge100_wf_s0", "G_t2_wf_s0"]:
    vocab = json.load(open(f"runs/{run}/vocab.json"))
    m = TwoTower(len(vocab), 512).to(dv)
    m.load_state_dict(torch.load(f"runs/{run}/model.pt", map_location=dv))
    m.eval()
    r = eval_4afc(m, emb, lut, ev, vocab, dv, seed=0)
    print(f"  {run:18s} dev117 {r['acc'] * 100:.1f} ({r['n_cats']}/117)")
