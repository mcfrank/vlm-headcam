"""Build the LEVANTE-vocab secondary eval: item table + image list from the (authoritative,
git-tracked) LEVANTE-bench item bank. Each test item is a target word + 4 candidate images
(correct + 3 distractors, 4AFC) with an IRT difficulty d. We mirror DevBench's contrastive eval:
score the target word against each of the 4 images and pick the best, then compare the model's
per-item accuracy to children's item difficulties.

Writes (to OUT):
  lev_vocab_items.parquet   item_uid, target_word, c0..c3 (concepts; c0 = correct), d
  lev_vocab_images.txt      unique image filenames to ship to ccn2
"""
import csv
import os
import sys
import pandas as pd

LEV = "/Users/mcfrank/Projects/levante-bench/data/assets/2026-02-22"
BANK = f"{LEV}/corpus/vocab/vocab-item-bank.csv"
IMGDIR = f"{LEV}/visual/vocab"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)


def canon(c):
    return c.lower().strip().replace(" ", "").replace("-", "")


stem = {canon(os.path.splitext(f)[0]): f for f in os.listdir(IMGDIR)}

rows = [r for r in csv.DictReader(open(BANK)) if r["trial_type"] == "test"]
items, needed = [], set()
for r in rows:
    concepts = [r["answer"].strip()] + [c.strip() for c in r["response_alternatives"].split(",")]
    if any(canon(c) not in stem for c in concepts):
        continue
    files = [stem[canon(c)] for c in concepts]
    needed.update(files)
    items.append({
        "item_uid": r["item_uid"],
        "target_word": r["item"].strip().lower(),
        "c0": files[0], "c1": files[1], "c2": files[2], "c3": files[3],  # c0 = correct
        "d": float(r["d"]) if r["d"] not in ("", "NA") else float("nan"),
    })

df = pd.DataFrame(items)
df.to_csv(f"{OUT}/lev_vocab_items.csv", index=False)
open(f"{OUT}/lev_vocab_images.txt", "w").write("\n".join(sorted(needed)))
print(f"{len(df)} test items | {len(needed)} unique images | {df.d.notna().sum()} with IRT d")
print(f"wrote {OUT}/lev_vocab_items.csv + lev_vocab_images.txt")
