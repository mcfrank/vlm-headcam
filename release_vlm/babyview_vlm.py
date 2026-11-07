"""Minimal loader + scorer for the BabyView word-learning VLMs (Frank et al.,
"Scaling cross-situational word learning on children's naturalistic experiences").

A model = a FROZEN DINOv3 image encoder (not included; see the model card) + the trained
pieces stored here: a projection of each image region into a 512-d space, a 512-d embedding
per vocabulary word, and a temperature. An utterance is the mean of its word embeddings; its
score against an image is the max, over the image's regions, of the cosine similarity.

Regions: the encoder's patch tokens average-pooled to a 4x4 grid (16 regions), optionally
preceded by the CLS token (17 regions). Models were TRAINED on the 16 grid regions; see the
model card for which variant reproduces the paper's evaluation numbers.

    import babyview_vlm as bv
    m = bv.load("path/to/repo", seed=0)                     # full-corpus model (seeds 0-4)
    m = bv.load("path/to/repo", seed=0, aligned=True)       # aligned-only model (seeds 0-2)
    regions = bv.regions_hf(images, "facebook/dinov3-vitl16-pretrain-lvd1689m")   # OTS encoders
    S = m.score(["dog", "ball"], regions)                   # [2 words, n images]
"""
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LEN = 16
_TOK = re.compile(r"[a-z]+")


class RegionMIL(nn.Module):
    def __init__(self, n_vocab, emb_dim, dim=512):
        super().__init__()
        self.vproj = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(0.1), nn.Linear(emb_dim, dim))
        self.word = nn.Embedding(n_vocab, dim, padding_idx=0)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("center_mu", torch.zeros(emb_dim))
        self.vocab = {}

    def tokenize(self, text):
        """Lowercase, keep runs of a-z, drop out-of-vocabulary words, keep the first 16."""
        return [self.vocab[t] for t in _TOK.findall(str(text).lower()) if t in self.vocab][:MAX_LEN]

    def enc_regions(self, v):                       # [B, R, emb_dim] -> [B, R, 512], unit norm
        return F.normalize(self.vproj(v - self.center_mu), dim=-1)

    def enc_text(self, texts):
        """[n] utterances -> [n, 512] unit-norm vectors; rows for all-OOV texts are NaN."""
        out = torch.full((len(texts), self.word.embedding_dim), float("nan"), device=self.word.weight.device)
        for i, t in enumerate(texts):
            ids = self.tokenize(t)
            if ids:
                out[i] = F.normalize(self.word(torch.tensor(ids, device=out.device)).mean(0), dim=-1)
        return out

    @torch.no_grad()
    def score(self, texts, regions):
        """Cosine score of each text against each image: max over regions. [n_texts, n_images]."""
        R = self.enc_regions(regions.to(self.word.weight.device, torch.float32))
        T = self.enc_text(texts)
        return torch.einsum("brd,td->tbr", R, T).max(-1).values


def load(repo, seed=0, aligned=False, device="cpu"):
    """Load a model from a local directory or a Hugging Face repo id (gated: request access on
    the model page first, then `hf auth login`). aligned=False: trained on the full corpus
    (seeds 0-4); aligned=True: trained only on referentially aligned pairs (seeds 0-2)."""
    from safetensors.torch import load_file
    sub = f"aligned/seed{seed}" if aligned else f"seed{seed}"
    root = Path(repo)
    if not root.exists():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(repo, allow_patterns=[f"{sub}/*"]))
    d = root / sub
    cfg = json.loads((d / "config.json").read_text())
    vocab = json.loads((d / "vocab.json").read_text())
    m = RegionMIL(len(vocab), cfg["emb_dim"])
    m.load_state_dict(load_file(d / "model.safetensors"))
    m.vocab, m.config = vocab, cfg
    return m.eval().to(device)


@torch.no_grad()
def regions_hf(images, encoder_id, include_cls=False, grid=4, device=None, batch=64):
    """Region features from a Hugging Face DINOv3 encoder (the off-the-shelf models), exactly
    as computed for the paper (src/embed_regions.py / src/embed_konkle.py). `images`: PIL images.
    Returns [n, 16 (or 17 with CLS), emb_dim]."""
    from transformers import AutoImageProcessor, AutoModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(encoder_id)
    enc = AutoModel.from_pretrained(encoder_id).eval().to(device)
    nreg = getattr(enc.config, "num_register_tokens", 0) or 0
    out = []
    for i in range(0, len(images), batch):
        h = enc(**proc(images=[im.convert("RGB") for im in images[i:i + batch]], return_tensors="pt").to(device)).last_hidden_state
        out.append(_pool(h[:, 0], h[:, 1 + nreg:], include_cls, grid))
    return torch.cat(out)


@torch.no_grad()
def regions_native(images, backbone, include_cls=False, grid=4, device="cuda", batch=96):
    """Region features from a NATIVE dinov3 backbone (the BabyView-trained encoders, as used
    in the paper): `backbone` = the teacher backbone built with the dinov3 fork
    (build_model(cfg.student, only_teacher=True, img_size=224) + the released state dict).
    224x224 resize, ImageNet normalisation, bf16 autocast — as in the paper."""
    import numpy as np
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([(torch.from_numpy(np.asarray(im.convert("RGB").resize((224, 224)), dtype=np.float32) / 255)
                          .permute(2, 0, 1) - mean) / std for im in images[i:i + batch]]).to(device)
        with torch.autocast("cuda", torch.bfloat16):
            o = backbone.forward_features(x)
        out.append(_pool(o["x_norm_clstoken"].float(), o["x_norm_patchtokens"].float(), include_cls, grid))
    return torch.cat(out)


def _pool(cls, patch, include_cls, grid):
    s = int(math.isqrt(patch.shape[1]))
    g = patch[:, :s * s].transpose(1, 2).reshape(len(patch), -1, s, s)
    pooled = F.adaptive_avg_pool2d(g, (grid, grid)).flatten(2).transpose(1, 2)
    feat = torch.cat([cls.unsqueeze(1), pooled], 1) if include_cls else pooled
    return feat.to(torch.float16).float().cpu()     # the paper's caches are float16
