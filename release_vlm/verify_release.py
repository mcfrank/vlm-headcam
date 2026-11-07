"""Verify the staged release (run on ccn2 from the vlm-headcam root):
  1. every seed's released files reproduce its recorded Konkle score exactly (paper eval code);
  2. the shipped loader's score() equals the paper model's region-max similarity;
  3. regions_hf / regions_native reproduce the paper's Konkle feature caches.
usage: python release_vlm/verify_release.py <staged_dir>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src"); sys.path.insert(0, "release_vlm")
from safetensors.torch import load_file
from train_region_mil import RegionMIL, eval_4afc_region, load_region_cache
import babyview_vlm as bv

ROOT = Path(sys.argv[1]); dev = "cuda" if torch.cuda.is_available() else "cpu"
EV = pd.read_parquet("manifests/eval_frames_konkle.parquet")
CACHE = {"ots-22m": "emb_ch8_eval/dinov3s_konkle", "ots-86m": "emb_enc_grid_eval/dinov3b_ots_konkle",
         "ots-304m": "emb_ch8_eval/dinov3l_grid4x4_konkle", "bv-22m": "emb_ch8_eval/vits_bv_konkle",
         "bv-86m": "emb_ch8_eval/vitb_bv_konkle", "bv-304m": "emb_ch8_eval/vitl_bv_konkle"}
bad = 0
for key, cache in CACHE.items():
    emb, lut = load_region_cache(cache)
    for sub in [f"seed{i}" for i in range(5)] + [f"aligned/seed{i}" for i in range(3)]:
        s = sub
        d = ROOT / f"babyview-vlm-{key}" / sub
        cfg = json.loads((d / "config.json").read_text()); vocab = json.loads((d / "vocab.json").read_text())
        m = RegionMIL(len(vocab), emb_dim=cfg["emb_dim"]).to(dev); m.load_state_dict(load_file(d / "model.safetensors")); m.eval()
        acc = round(100 * eval_4afc_region(m, emb, lut, EV, vocab, dev), 2)
        ok = acc == cfg["konkle_test60_4afc"]; bad += not ok
        line = f"{key} {sub}: released {acc:.2f} vs recorded {cfg['konkle_test60_4afc']:.2f} {'OK' if ok else 'MISMATCH'}"
        if sub == "seed0":   # 2. loader equivalence on real Konkle features
            L = bv.load(ROOT / f"babyview-vlm-{key}", seed=0, device=dev)
            V = torch.from_numpy(np.asarray(emb[:40], dtype=np.float32))
            words = ["dog", "ball", "airplane", "spoon", "cat"]
            S1 = L.score(words, V)
            with torch.no_grad():
                Rp = m.enc_regions(V.to(dev)); T = []
                for w in words:
                    ids = [vocab[w]]; t = torch.zeros(1, 16, dtype=torch.long, device=dev); t[0, 0] = ids[0]
                    T.append(m.enc_text(t, torch.tensor([1], device=dev)))
                S2 = torch.einsum("brd,td->tbr", Rp, torch.cat(T)).max(-1).values
            diff = float((S1 - S2).abs().max()); bad += diff > 1e-5
            line += f" | loader vs paper score max|diff| {diff:.1e}"
        print(line, flush=True)
# CLS sensitivity: score the paper's 17-region caches with the CLS row dropped (grid only)
for key in ["ots-22m", "ots-304m", "bv-22m", "bv-86m", "bv-304m"]:
    emb, lut = load_region_cache(CACHE[key]); grid_only = np.ascontiguousarray(np.asarray(emb)[:, 1:])
    diffs = []
    for i in range(5):
        d = ROOT / f"babyview-vlm-{key}" / f"seed{i}"; cfg = json.loads((d / "config.json").read_text())
        vocab = json.loads((d / "vocab.json").read_text())
        m = RegionMIL(len(vocab), emb_dim=cfg["emb_dim"]).to(dev); m.load_state_dict(load_file(d / "model.safetensors")); m.eval()
        diffs.append(100 * eval_4afc_region(m, grid_only, lut, EV, vocab, dev) - cfg["konkle_test60_4afc"])
    print(f"CLS sensitivity {key}: grid-only minus with-CLS = {np.mean(diffs):+.2f} (per seed {np.round(diffs, 2).tolist()})", flush=True)
# 3a. HF feature extraction vs the paper cache (OTS-304M, CLS + grid)
from PIL import Image
R = Path("/ccn2b/dataset/babyview/eval_assets"); man = pd.read_parquet(R / "manifests/konkle_manifest.parquet").iloc[:64]
ims = [Image.open(R / p) for p in man.path]
emb, lut = load_region_cache("emb_ch8_eval/dinov3l_grid4x4_konkle")
ref = torch.from_numpy(np.asarray(emb[[lut[f"{v}|{f}"] for v, f in zip(man.video_id, man.frame_idx)]], dtype=np.float32))
got = bv.regions_hf(ims, "facebook/dinov3-vitl16-pretrain-lvd1689m", include_cls=True, batch=128)
print(f"regions_hf vs paper cache (64 Konkle images): exact {torch.equal(got, ref)} | max|diff| {float((got-ref).abs().max()):.2e}")
