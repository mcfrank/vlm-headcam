"""Eval whole-frame (TwoTower) models on Konkle by using the CLS token of the region cache."""
import sys, json, os
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from train import TwoTower, eval_4afc
from common import load_emb_cache
W = "/data2/mcfrank/vlm-headcam"; dev = "cuda" if torch.cuda.is_available() else "cpu"

# build a CLS-only Konkle cache from emb_konkle (row 0 of the region grid)
cls_dir = f"{W}/emb_konkle_cls"
if not os.path.exists(f"{cls_dir}/emb.f16.npy"):
    os.makedirs(cls_dir, exist_ok=True)
    reg = np.load(f"{W}/emb_konkle/emb.f16.npy")           # [N,17,768]
    np.save(f"{cls_dir}/emb.f16.npy", reg[:, 0, :])        # [N,768] CLS
    import shutil; shutil.copy(f"{W}/emb_konkle/index.parquet", f"{cls_dir}/index.parquet")
    print("built emb_konkle_cls", reg[:, 0, :].shape)

emb, lut = load_emb_cache(cls_dir)
ev = pd.read_parquet(f"{W}/manifests/eval_frames_konkle.parquet")
for run in sys.argv[1:]:
    try:
        vocab = json.load(open(f"{W}/runs/{run}/vocab.json"))
        m = TwoTower(len(vocab), 512).to(dev)
        m.load_state_dict(torch.load(f"{W}/runs/{run}/model.pt", map_location=dev)); m.eval()
        res = eval_4afc(m, emb, lut, ev, vocab, dev, seed=0)
        print(f"  {run:26s} Konkle 4AFC={res['acc']*100:.1f}  (cats {res['n_cats']}/{ev.category.nunique()})")
    except Exception as e:
        print(f"  {run}: ERR {e}")
