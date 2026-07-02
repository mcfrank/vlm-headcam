#!/usr/bin/env bash
# Publish the book to Quarto Pub (a PUBLIC url) with the human-subjects frames REMOVED.
#
# The figures in figures/frames/ show participants' faces and must never reach a public surface.
# This script backs them up (by COPY, so the originals are never at risk), substitutes small
# "withheld" placeholders, does a clean render, runs a SAFETY GATE that refuses to publish if a
# real frame leaked through, publishes, and restores the originals. If any step dies, the backup
# in figures/frames_hold/ still holds the originals.
#
#   ./publish-public.sh            # render frame-free, safety-check, publish (first run: browser auth)
#   DRY_RUN=1 ./publish-public.sh  # everything except the publish (safe to test)
set -euo pipefail
cd "$(dirname "$0")"
command -v quarto >/dev/null || { echo "quarto not found"; exit 1; }

FRAMES=figures/frames
HOLD=figures/frames_hold          # gitignored; a safety COPY of the originals

restore() {                        # put originals back from the backup copy
  if [ -d "$HOLD" ]; then rm -rf "$FRAMES"; mv "$HOLD" "$FRAMES"; echo "restored $FRAMES/ from backup"; fi
}
trap restore EXIT

# 1) back up the real frames by COPY (originals stay in place until safely copied)
if [ -d "$FRAMES" ] && [ -n "$(ls -A "$FRAMES" 2>/dev/null)" ]; then
  rm -rf "$HOLD"; cp -R "$FRAMES" "$HOLD"
else
  echo "note: no frames in $FRAMES (already frame-free) — nothing to hide."
fi

# 2) overwrite each real frame with a small "withheld" placeholder so the public book has no
#    broken images. Best-effort renderer (PIL -> matplotlib -> empty file); all are safe.
if [ -d "$HOLD" ]; then
  for f in "$FRAMES"/*; do
    [ -e "$f" ] || continue
    python3 - "$f" <<'PY' 2>/dev/null || : > "$f"
import sys
msg = "Example frame withheld (human subjects) - see the lab version."
try:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (900, 320), (238, 238, 238))
    ImageDraw.Draw(im).text((40, 150), msg, fill=(90, 90, 90))
    im.save(sys.argv[1])
except Exception:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 3.2)); fig.text(0.5, 0.5, msg, ha="center", color="#5a5a5a")
    fig.savefig(sys.argv[1]); plt.close(fig)
PY
  done
fi

# 3) clean render — wipe _book and caches so no previously-rendered real frame can leak
rm -rf _book .quarto _freeze
quarto render

# 4) SAFETY GATE: real headcam frames are large; placeholders are tiny. Abort on any big frame.
fail=0
for img in _book/figures/frames/* ; do
  [ -e "$img" ] || continue
  sz=$(stat -f%z "$img" 2>/dev/null || stat -c%s "$img" 2>/dev/null || echo 0)
  [ "$sz" -gt 60000 ] && { echo "ABORT: $img is ${sz} bytes — looks like a REAL frame."; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "Safety gate failed — NOT publishing. Originals restored on exit."; exit 1; }
echo "Safety gate passed: only placeholders in _book/figures/frames/."

# 5) publish
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 — skipping publish. Frame-free book is in _book/; originals will be restored now."
  exit 0
fi
quarto publish quarto-pub --no-render      # first run: browser auth + writes _publish.yml
echo
echo "Published. If _publish.yml was just created, commit it so future publishes reuse the URL:"
echo "  git add _publish.yml && git commit -m 'Add Quarto Pub target'"
