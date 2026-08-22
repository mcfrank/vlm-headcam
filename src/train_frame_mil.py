"""Frame-MIL: an utterance matches its best (frame, region) over a ±W-second window, not just
the midpoint frame. Extends region-MIL's max-over-cells to a max over frames × cells by stacking
the window frames' region grids and letting the same RegionMIL.sims max over all of them.

Cells of the region×frame 2×2:
  region-only  = region-MIL baseline (window 0)                    -> already have (G_base_mil)
  whole-frame-only = --cls-only --window 0                          -> ~ the no-MIL baseline
  frame-only   = --cls-only --window 2  (max over frames' CLS)
  region+frame = --window 2             (max over frames × regions)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from common import frame_key, save_json
from train import build_vocab, encode
from train_region_mil import RegionMIL, eval_4afc_region, collate, load_region_cache


def load_multi(dirs):
    """Union several region caches into one key -> (cache_idx, row) lut; caches stay mmap'd."""
    caches, lut = [], {}
    for ci, d in enumerate(dirs):
        idx = pd.read_parquet(Path(d) / "index.parquet")
        caches.append(np.load(Path(d) / "emb.f16.npy", mmap_mode="r"))
        for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row):
            lut[frame_key(v, int(f))] = (ci, int(r))
    return caches, lut


class FramePairs(torch.utils.data.Dataset):
    def __init__(self, caches, lut, man, vocab, window, cls_only, max_len=16):
        self.caches, self.window, self.cls_only, self.max_len = caches, window, cls_only, max_len
        self.Fmax = 2 * window + 1
        self.locs, self.ids = [], []
        for r in man.itertuples(index=False):
            toks = encode(r.text, vocab, max_len)
            if not toks:
                continue
            fi = int(r.frame_idx)
            here = [lut[frame_key(r.video_id, f)] for f in range(fi - window, fi + window + 1)
                    if frame_key(r.video_id, f) in lut]
            if not here:
                continue
            self.locs.append(here); self.ids.append(toks)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        grids = [np.asarray(self.caches[ci][row], dtype=np.float32) for ci, row in self.locs[i]]
        while len(grids) < self.Fmax:                 # pad by repeat (max-pool ignores dups)
            grids.append(grids[0])
        v = np.stack(grids)                            # [Fmax, R, 768]
        if self.cls_only:
            v = v[:, 0:1, :]
        v = v.reshape(-1, v.shape[-1])                 # [Fmax*R', 768]
        t = torch.zeros(self.max_len, dtype=torch.long)
        t[:len(self.ids[i])] = torch.tensor(self.ids[i])
        return i, torch.from_numpy(v), t, len(self.ids[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--eval-frames", required=True)
    # Model selection on a DEV split (Konkle dev-117) and reporting on test-60 removes the
    # best-epoch-on-test optimism (+1.4 pts on average, but 0.4-3.5 depending on condition, which
    # distorts comparisons). If omitted, falls back to the legacy best-on-test behaviour.
    ap.add_argument("--dev-cache", default=None)
    ap.add_argument("--dev-frames", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--cls-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-dim", type=int, default=None, help="encoder feature dim (default: cache dim)")
    ap.add_argument("--center", action="store_true", help="subtract train dataset-mean before projection (anisotropy fix)")
    # A cache that does not cover the manifest silently shrinks the training set — this is exactly
    # how the 2026-08-22 phase-5 runs trained on 9% of their manifests without anyone noticing.
    # Coverage is always reported; pass --min-coverage to make a shortfall fatal.
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="abort if the caches cover less than this fraction of the manifest")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    man = pd.read_parquet(a.manifest)
    vocab = build_vocab(man.text, 5)
    caches, lut = load_multi(a.caches)
    emb_dim = a.emb_dim or caches[0].shape[-1]
    ds = FramePairs(caches, lut, man, vocab, a.window, a.cls_only)
    cov = len(ds) / max(len(man), 1)
    print(f"pairs {len(ds)} | manifest {len(man)} | coverage {100*cov:.1f}% | vocab {len(vocab)} "
          f"| window +-{a.window} | cls_only {a.cls_only} | emb_dim {emb_dim}", flush=True)
    if cov < 0.999:
        print(f"  NOTE: {len(man) - len(ds):,} pairs dropped — their frames are not in the caches",
              flush=True)
    if cov < a.min_coverage:
        raise SystemExit(f"ABORT: cache coverage {100*cov:.1f}% < required {100*a.min_coverage:.0f}%. "
                         f"The cache does not span this manifest — check you are using the right one.")
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     collate_fn=collate, num_workers=4)
    m = RegionMIL(len(vocab), emb_dim=emb_dim).to(dev)
    if a.center:
        mu = np.asarray(caches[0]).reshape(-1, emb_dim).mean(0)
        m.center_mu.copy_(torch.from_numpy(mu.astype(np.float32)).to(dev))
        print(f"centered: subtracted train mean |mu|={np.linalg.norm(mu):.2f}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.1)
    ev = pd.read_parquet(a.eval_frames); ecache, elut = load_region_cache(a.eval_cache)
    use_dev = bool(a.dev_cache and a.dev_frames)
    if use_dev:
        dv = pd.read_parquet(a.dev_frames); dcache, dlut = load_region_cache(a.dev_cache)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    hist, best_sel, best_state, best_ep = [], -1.0, None, -1
    for ep in range(a.epochs):
        m.train()
        for _, v, t, n in dl:
            v, t, n = v.to(dev), t.to(dev), n.to(dev)
            w = torch.ones(len(v), device=dev)
            loss = m.forward_loss(v, t, n, w)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = eval_4afc_region(m, ecache, elut, ev, vocab, dev)
        dacc = eval_4afc_region(m, dcache, dlut, dv, vocab, dev) if use_dev else None
        sel = dacc if use_dev else acc            # what we select the epoch on
        if sel > best_sel:
            best_sel, best_ep = sel, ep
            best_state = {k: v_.detach().cpu().clone() for k, v_ in m.state_dict().items()}
        rec = {"ep": ep, "acc": round(acc, 4)}
        if use_dev:
            rec["dev"] = round(dacc, 4)
        hist.append(rec)
        print(rec, flush=True)

    # report the TEST accuracy at the DEV-selected epoch (or best-on-test in legacy mode)
    reported = hist[best_ep]["acc"]
    if best_state is not None:
        m.load_state_dict(best_state)
    torch.save(m.state_dict(), Path(a.out) / "model.pt")
    save_json(vocab, Path(a.out) / "vocab.json")
    # metrics.json: the run's own machine-readable record, so no number ever has to be
    # recovered from stdout again (see notes/PROVENANCE.md).
    save_json({"run": str(Path(a.out).name), "seed": a.seed, "manifest": a.manifest,
               "caches": a.caches, "eval_frames": a.eval_frames, "window": a.window,
               "cls_only": a.cls_only, "center": a.center, "emb_dim": int(emb_dim),
               "epochs": a.epochs, "n_pairs": len(ds), "vocab": len(vocab),
               "selection": "dev" if use_dev else "test",
               "selected_epoch": best_ep, "reported_test_acc": round(100 * reported, 3),
               "best_test_acc": round(100 * max(h["acc"] for h in hist), 3),
               "final_test_acc": round(100 * hist[-1]["acc"], 3),
               "history": hist}, Path(a.out) / "metrics.json")
    print(f"DONE {a.out} selected_ep {best_ep} ({'dev' if use_dev else 'test'}) "
          f"reported {reported:.4f} best {max(h['acc'] for h in hist):.4f}", flush=True)


if __name__ == "__main__":
    main()
