"""Scrape EVERY training run into one canonical results table — the single source of truth
for every model number in the book and paper.

Two log formats exist in this project's history:
  A. runs/<name>/log.json      list of {epoch|round, acc_4afc, ...}   (train_region_mil, train.py)
  B. logs/<name>.log           lines "{'ep': N, 'acc': X}" + "DONE <path> best X"  (train_frame_mil,
                               train_arch, train_caption — the clean-rig era)
Format B is also how the ch8/ch9 (encoder/architecture) runs survive: their run dirs hold only
model.pt, so the text logs restored from Oak are the only metric record.

usage: python src/scrape_runs.py            # writes results/runs.parquet
"""
import ast, glob, json, os, re
import pandas as pd

W = "/data2/mcfrank/vlm-headcam"
OAK_LOGS = "/data2/mcfrank/tmp/home-runs"   # log.json-less ch8/ch9 logs restored from Oak

rows = []


def add(run, accs, store, source, n_pairs=None, n_cats=None, best_reported=None):
    if not accs:
        return
    m = re.search(r"_s(\d+)$", run)
    rows.append(dict(
        run=run, family=re.sub(r"_s\d+$", "", run),
        seed=int(m.group(1)) if m else None,
        best_acc=round(100 * max(accs), 2),
        final_acc=round(100 * accs[-1], 2),
        best_epoch=int(accs.index(max(accs))), n_epochs=len(accs),
        n_pairs=n_pairs, n_eval_cats=n_cats,
        best_reported=None if best_reported is None else round(100 * best_reported, 2),
        store=store, source=source,
    ))


# ---- format A: runs/*/log.json -------------------------------------------------
for lj in sorted(glob.glob(f"{W}/runs/*/log.json")):
    run = os.path.basename(os.path.dirname(lj))
    try:
        hist = json.load(open(lj))
    except Exception as e:
        print("SKIP", lj, e); continue
    if not isinstance(hist, list):
        continue
    accs = [h["acc_4afc"] for h in hist if isinstance(h, dict) and h.get("acc_4afc") is not None]
    h0 = hist[0] if hist and isinstance(hist[0], dict) else {}
    add(run, accs, "data2", f"runs/{run}/log.json", h0.get("eff"), h0.get("n_eval_cats"))

# ---- format B: text logs (data2 + the two Oak-restored trees) ------------------
EP = re.compile(r"\{'ep':\s*\d+,\s*'acc':\s*([0-9.]+)\}")
DONE = re.compile(r"DONE\s+(\S+)\s+best\s+([0-9.]+)")
PAIRS = re.compile(r"^pairs\s+(\d+)")

for store, pat in [("data2", f"{W}/logs/*.log"),
                   ("oak:vlm_enc", f"{OAK_LOGS}/vlm_enc-logs/*.log"),
                   ("oak:vlm_arch", f"{OAK_LOGS}/vlm_arch-logs/*.log")]:
    for lf in sorted(glob.glob(pat)):
        txt = open(lf, errors="ignore").read()
        accs = [float(x) for x in EP.findall(txt)]
        d = DONE.search(txt)
        if not accs and not d:
            continue
        # the DONE line names the true run dir; fall back to the log's filename
        run = os.path.basename(d.group(1)) if d else os.path.basename(lf)[:-4]
        p = PAIRS.search(txt)
        add(run, accs, store, os.path.relpath(lf, os.path.dirname(pat.rstrip("*.log"))),
            int(p.group(1)) if p else None, None,
            float(d.group(2)) if d else None)

df = pd.DataFrame(rows)
# a run can appear in both a log.json and a text log — keep the richer (log.json) row
df = df.sort_values(["run", "n_epochs"]).drop_duplicates("run", keep="last")
df = df.sort_values(["family", "seed"]).reset_index(drop=True)
os.makedirs(f"{W}/results", exist_ok=True)
df.to_parquet(f"{W}/results/runs.parquet", index=False)
print(f"{len(df)} runs | {df.family.nunique()} families | stores: {dict(df.store.value_counts())}")
print("\nsanity — families the book cites:")
for fam in ["G_base_mil_full", "G_sc_scale_rand_911000", "G_sc_scale_align_10000",
            "G_t15_filtnat_mil", "E_dinov3b_ots_wf", "E_dinov3b_ots_grid"]:
    s = df[df.family == fam]
    if len(s):
        print(f"  {fam:28s} n={len(s)} best={s.best_acc.mean():.1f} ± {s.best_acc.std():.1f}")
    else:
        print(f"  {fam:28s} MISSING")
