"""P1 — self-bootstrapped alignment filter (no external CLIP in training).

The learner starts with NO grounding. It warms up on ALL pairs, then iterates an
EM-style loop: (E) score each pair by the model's OWN cross-modal cosine similarity
and fit a 2-component 1-D Gaussian mixture to those scores to get a soft "aligned"
weight per pair; (M) keep training with the InfoNCE positive term weighted by that
belief. Better word meanings -> sharper alignment beliefs -> cleaner training. This is
the neural analogue of Frank-Goodman-Tenenbaum (2009) joint inference of lexicon +
referential intent.

CLIP scores in the manifest are NEVER used for training — only logged as a held-out
yardstick (does the endogenous weight recover the CLIP-aligned pairs?).
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from common import MANIFEST_DIR, EMB_DIR, load_emb_cache, tokenize, save_json, frame_key
from train import build_vocab, encode, eval_4afc, TwoTower


class BootPairs(torch.utils.data.Dataset):
    """Keeps pair order and the (held-out, unused-for-training) clip score per kept pair."""
    def __init__(self, emb, lut, manifest, vocab, max_len=16):
        self.embrows, self.ids, self.clip = [], [], []
        has_key = "key" in manifest.columns
        for r in manifest.itertuples(index=False):
            key = r.key if has_key else frame_key(r.video_id, r.frame_idx)
            if key not in lut:
                continue
            toks = encode(r.text, vocab, max_len)
            if not toks:
                continue
            self.embrows.append(lut[key]); self.ids.append(toks)
            self.clip.append(float(getattr(r, "clip_score_max", np.nan)))
        self.emb = emb; self.max_len = max_len
        self.clip = np.array(self.clip)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        v = torch.from_numpy(np.asarray(self.emb[self.embrows[i]], dtype=np.float32))
        t = torch.zeros(self.max_len, dtype=torch.long)
        t[:len(self.ids[i])] = torch.tensor(self.ids[i])
        return i, v, t, len(self.ids[i])


def collate_boot(b):
    idx = torch.tensor([x[0] for x in b])
    v = torch.stack([x[1] for x in b]); t = torch.stack([x[2] for x in b])
    n = torch.tensor([x[3] for x in b])
    return idx, v, t, n


def gmm2_posterior(x, iters=100):
    """1-D two-Gaussian EM; return P(high-mean component | x) as the 'aligned' weight."""
    x = np.asarray(x, float)
    mu = np.percentile(x, [25, 75]).astype(float)
    var = np.full(2, x.var() + 1e-6)
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        g = np.stack([pi[k]/np.sqrt(2*np.pi*var[k]) *
                      np.exp(-(x-mu[k])**2/(2*var[k])) for k in range(2)])
        g += 1e-12
        r = g / g.sum(0)
        nk = r.sum(1)
        pi = nk / len(x)
        mu = (r*x).sum(1) / nk
        var = (r*(x-mu[:, None])**2).sum(1) / nk + 1e-6
    hi = int(np.argmax(mu))
    return r[hi]


def weighted_infonce(model, v, t, n, w):
    iv, tv = model.encode_image(v), model.encode_text(t, n)
    scale = model.logit_scale.clamp(max=np.log(100)).exp()
    logits = scale * iv @ tv.t()
    labels = torch.arange(len(v), device=v.device)
    li = F.cross_entropy(logits, labels, reduction="none")
    lt = F.cross_entropy(logits.t(), labels, reduction="none")
    return (0.5 * w * (li + lt)).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def pair_scores(model, ds, dev, bs=1024):
    model.eval()
    s = np.zeros(len(ds), dtype=np.float32)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_boot)
    for idx, v, t, n in dl:
        iv = model.encode_image(v.to(dev)); tv = model.encode_text(t.to(dev), n.to(dev))
        s[idx.numpy()] = (iv * tv).sum(-1).cpu().numpy()   # cosine (both normalized)
    return s


def yardstick(w, clip):
    """How well does the endogenous weight recover the held-out CLIP-aligned pairs?"""
    ok = ~np.isnan(clip)
    w, clip = w[ok], clip[ok]
    if len(w) < 50 or clip.std() == 0:
        return {}
    # Spearman via rank correlation
    rw = pd.Series(w).rank().to_numpy(); rc = pd.Series(clip).rank().to_numpy()
    rho = float(np.corrcoef(rw, rc)[0, 1])
    truth = clip > 0.24
    sel = w > 0.5
    prec = float((truth & sel).sum() / max(1, sel.sum()))
    rec = float((truth & sel).sum() / max(1, truth.sum()))
    return {"spearman_w_clip": round(rho, 3), "sel_frac": round(float(sel.mean()), 3),
            "precision_vs_clip": round(prec, 3), "recall_vs_clip": round(rec, 3),
            "base_rate_clip>0.24": round(float(truth.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--eval-frames", default=str(MANIFEST_DIR / "eval_frames.parquet"))
    ap.add_argument("--emb-dir", default=str(EMB_DIR))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=3, help="unweighted warm-up epochs")
    ap.add_argument("--rounds", type=int, default=6, help="EM rounds after warm-up")
    ap.add_argument("--epochs-per-round", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--floor", type=float, default=0.05, help="min weight (keep some exploration)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    emb, lut = load_emb_cache(args.emb_dir)
    man = pd.read_parquet(args.manifest)
    ev = pd.read_parquet(args.eval_frames)
    vocab = build_vocab(man.text, args.min_freq)
    ds = BootPairs(emb, lut, man, vocab)
    print(f"pairs {len(ds)} | vocab {len(vocab)} | clip base-rate>0.24 "
          f"{np.nanmean(ds.clip>0.24):.3f}", flush=True)

    model = TwoTower(len(vocab), args.dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    W = torch.ones(len(ds), device=dev)   # all pairs equal during warm-up
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     drop_last=True, collate_fn=collate_boot)
    log = []

    def run_epochs(k, phase, rnd):
        for _ in range(k):
            model.train()
            for idx, v, t, n in dl:
                loss = weighted_infonce(model, v.to(dev), t.to(dev), n.to(dev), W[idx.to(dev)])
                opt.zero_grad(); loss.backward(); opt.step()
        acc = eval_4afc(model, emb, lut, ev, vocab, dev, seed=args.seed)["acc"]
        rec = {"phase": phase, "round": rnd, "acc_4afc": round(acc, 4),
               "eff_pairs": int(W.gt(0.5).sum().item())}
        log.append(rec); print(rec, flush=True)
        return acc

    # warm-up (unweighted) — let the memorization effect surface aligned pairs
    run_epochs(args.warmup, "warmup", -1)

    # EM rounds: re-estimate the endogenous alignment weight, then train on it
    for r in range(args.rounds):
        s = pair_scores(model, ds, dev)
        w = gmm2_posterior(s)
        w = np.clip(w, args.floor, 1.0)
        W = torch.tensor(w, dtype=torch.float32, device=dev)
        acc = run_epochs(args.epochs_per_round, "boot", r)
        y = yardstick(w, ds.clip)
        log[-1].update(y)
        print("  yardstick:", y, flush=True)

    save_json(log, out / "log.json")
    # persist final weights + clip for later analysis
    np.savez(out / "weights.npz", w=w, clip=ds.clip, s=s)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
