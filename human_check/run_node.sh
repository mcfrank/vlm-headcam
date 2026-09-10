#!/usr/bin/env bash
# Run the app on ccn2-14 (ssh-tunnel deployment). Lab mates without a Google-account grant can use
#   ssh -L 8501:localhost:8501 ccn2-14   then open http://localhost:8501
# Binds to localhost only; data + responses stay under /data2/mcfrank/gemini_check.
set -euo pipefail
ROOT=/data2/mcfrank/gemini_check
HERE="$(cd "$(dirname "$0")" && pwd)"
pkill -u "$USER" -f "[u]vicorn --app-dir .* --port 8501" 2>/dev/null || true   # bracket: don't match this shell
cd "$ROOT"
(DATA_DIR=$ROOT/data RATERS_PER_ITEM=3 nohup "$ROOT/venv/bin/uvicorn" --app-dir "$HERE/app" main:app \
    --host 127.0.0.1 --port 8501 > "$ROOT/app.log" 2>&1 &)
sleep 2; tail -3 "$ROOT/app.log"
