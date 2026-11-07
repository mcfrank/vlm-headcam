"""Upload the staged word-learning VLMs to Hugging Face as GATED (manual approval) repos and
group them in a collection. Run on ccn2 after `hf auth login` with a WRITE token for the
account that owns the BabyView DINO repos:
    python release_vlm/upload_release.py /data2/mcfrank/hf_vlm_release [--owner mcxfrank]
Repos are created PRIVATE, gated, then uploaded; nothing becomes visible until --publish.
    python release_vlm/upload_release.py /data2/mcfrank/hf_vlm_release --publish
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("staged")
ap.add_argument("--owner", default="mcxfrank")
ap.add_argument("--publish", action="store_true", help="flip the (already uploaded) repos from private to public")
a = ap.parse_args()
api = HfApi()
print("logged in as", api.whoami()["name"])
repos = sorted(p.name for p in Path(a.staged).iterdir() if p.name.startswith("babyview-vlm-"))
assert len(repos) == 6, repos
for name in repos:
    rid = f"{a.owner}/{name}"
    if a.publish:
        api.update_repo_settings(rid, private=False)
        print("public:", rid, "| gated:", api.model_info(rid).gated)
        continue
    api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
    api.update_repo_settings(rid, gated="manual")
    api.upload_folder(repo_id=rid, folder_path=str(Path(a.staged) / name),
                      commit_message="Release full-corpus word-learning VLM (5 seeds)")
    files = api.list_repo_files(rid)
    assert all(f"seed{s}/model.safetensors" in files for s in range(5)) and "README.md" in files, files
    print("uploaded (private, gated=manual):", rid, f"{len(files)} files")
if not a.publish:
    col = api.create_collection(title="BabyView word-learning VLMs", namespace=a.owner, private=True, exists_ok=True,
                                description="Cross-situational word learners trained on BabyView (Frank et al.). Gated: BabyView/Databrary access required.")
    for name in repos:
        api.add_collection_item(col.slug, item_id=f"{a.owner}/{name}", item_type="model", exists_ok=True)
    print("collection:", col.slug, "(private until published in the web UI)")
