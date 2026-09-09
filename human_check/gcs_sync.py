"""Move the app data between ccn2 and the private GCS bucket that backs the Cloud Run deployment.
Runs on ccn2 with the Vertex service-account key (which deploy.sh grants objectAdmin on the bucket):

    GOOGLE_APPLICATION_CREDENTIALS=~/.secrets/vlm-headcam-sa.json \
      venv/bin/python human_check/gcs_sync.py push --data /data2/mcfrank/gemini_check/data --bucket BUCKET
    ... pull --data /data2/mcfrank/gemini_check/data --bucket BUCKET   # responses/ back for analysis

push uploads frames/*.jpg + items.json (skips objects already present); pull downloads responses/.
"""
import argparse
from pathlib import Path

from google.cloud import storage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--bucket", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    b = storage.Client().bucket(args.bucket)
    if args.cmd == "push":
        have = {o.name for o in b.list_blobs(prefix="frames/")}
        paths = [data / "items.json"] + sorted((data / "frames").glob("*.jpg"))
        n = 0
        for p in paths:
            rel = str(p.relative_to(data))
            if rel in have and p.name != "items.json":
                continue
            b.blob(rel).upload_from_filename(p)
            n += 1
        print(f"uploaded {n} objects to gs://{args.bucket}")
    else:
        n = 0
        for o in b.list_blobs(prefix="responses/"):
            dst = data / o.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            o.download_to_filename(dst)
            n += 1
        print(f"downloaded {n} responses to {data / 'responses'}")


if __name__ == "__main__":
    main()
