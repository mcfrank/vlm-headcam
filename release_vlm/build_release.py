"""Stage the six full-corpus word-learning VLMs (all five seeds each, plus the aligned-only
models, three seeds each) for Hugging Face.
Run on ccn2 from the vlm-headcam root:  python release_vlm/build_release.py <out_dir>
Writes <out_dir>/<repo_name>/{README.md, babyview_vlm.py, seed0..4/, aligned/seed0..2/}, each
seed dir holding model.safetensors, vocab.json, config.json. Nothing is uploaded."""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from safetensors.torch import save_file

OUT = Path(sys.argv[1]); HERE = Path(__file__).resolve().parent
ENC = [  # tag, label, repo name, params, encoder id, paper Konkle eval cache
 ("dinov3s", "OTS-22M", "babyview-vlm-ots-22m", "22M", "facebook/dinov3-vits16-pretrain-lvd1689m", "emb_ch8_eval/dinov3s_konkle"),
 ("dinov3b", "OTS-86M", "babyview-vlm-ots-86m", "86M", "facebook/dinov3-vitb16-pretrain-lvd1689m", "emb_enc_grid_eval/dinov3b_ots_konkle"),
 ("dinov3l", "OTS-304M", "babyview-vlm-ots-304m", "304M", "facebook/dinov3-vitl16-pretrain-lvd1689m", "emb_ch8_eval/dinov3l_grid4x4_konkle"),
 ("vits_bv", "BV-22M", "babyview-vlm-bv-22m", "22M", "mcxfrank/babyview-dino-vits16", "emb_ch8_eval/vits_bv_konkle"),
 ("vitb_bv", "BV-86M", "babyview-vlm-bv-86m", "86M", "mcxfrank/babyview-dino-vitb16", "emb_ch8_eval/vitb_bv_konkle"),
 ("vitl_bv", "BV-304M", "babyview-vlm-bv-304m", "304M", "mcxfrank/babyview-dino-vitl16", "emb_ch8_eval/vitl_bv_konkle"),
]
grid = pd.read_csv("results/encoder_grid.csv")
def g(tag, measure):
    r = grid[(grid.encoder == tag) & (grid.measure == measure)].iloc[0]; return f"{r['mean']:.1f} ± {r['sd']:.1f}"
table = "\n".join(f"| {lab} | `{eid}` | {g(t, 'Konkle full')} | {g(t, 'aligned-only 172k')} | {g(t, 'in-domain full')} | {g(t, 'LEVANTE full (fair)')} |"
                  for t, lab, _, _, eid, _ in ENC)
template = (HERE / "README_template.md").read_text()
for tag, label, name, params, enc_id, ev in ENC:
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    R_eval = np.load(f"{ev}/emb.f16.npy", mmap_mode="r").shape[1]
    seeds, aseeds = [], []
    for s, cond, sub, rows in [(s, "base", f"seed{s}", seeds) for s in range(5)] + \
                              [(s, "alignedonly", f"aligned/seed{s}", aseeds) for s in range(3)]:
        rd = Path(f"runs/F_{tag}_{cond}_s{s}"); m = json.load(open(rd / "metrics.json"))
        sd = {k: v.contiguous() for k, v in torch.load(rd / "model.pt", map_location="cpu").items()}
        assert float(sd["center_mu"].abs().sum()) == 0 and m["window"] == 0 and not m["cls_only"]
        sdir = d / sub; sdir.mkdir(parents=True, exist_ok=True)
        save_file(sd, sdir / "model.safetensors")
        shutil.copy(rd / "vocab.json", sdir / "vocab.json")
        cfg = dict(model=label, encoder=enc_id, encoder_params=params, emb_dim=int(sd["vproj.2.weight"].shape[1]),
                   text_dim=512, vocab_size=int(sd["word.weight"].shape[0]), train_regions="4x4 grid (16)",
                   paper_eval_regions=("CLS + 4x4 grid (17)" if R_eval == 17 else "4x4 grid (16)"),
                   training_data=("full corpus" if cond == "base" else
                                  "referentially aligned pairs only (Gemini alignment >= 50)"),
                   seed=s, training_pairs=m["n_pairs"], epochs=m["epochs"], selected_epoch=m["selected_epoch"],
                   konkle_test60_4afc=m["reported_test_acc"], selection="dev-117 (Konkle) epoch selection")
        (sdir / "config.json").write_text(json.dumps(cfg, indent=1))
        rows.append(f"| `{sub}` | {m['n_pairs']:,} | {len(json.load(open(rd / 'vocab.json'))):,} | {m['reported_test_acc']:.2f} | {m['selected_epoch']} |")
    shutil.copy(HERE / "babyview_vlm.py", d / "babyview_vlm.py")
    (d / "README.md").write_text(template.format(
        name=name, label=label, params=params, encoder=enc_id, emb_dim=cfg["emb_dim"],
        regime=("off-the-shelf DINOv3 (internet-pretrained, Meta)" if tag.startswith("dinov3")
                else "DINOv3 trained from scratch on BabyView frames"),
        encoder_link=(f"https://huggingface.co/{enc_id}"),
        include_cls=("True" if R_eval == 17 else "False"),
        loader=("regions = bv.regions_hf(images, \"" + enc_id + "\", include_cls=" + ("True" if R_eval == 17 else "False") + ")"
                if tag.startswith("dinov3") else
                "regions = bv.regions_native(images, backbone, include_cls=True)  # backbone: see the encoder's model card"),
        seed_table="\n".join(seeds + aseeds), table=table))
    print(name, "| eval regions", R_eval, "| seeds 0-4 staged")
