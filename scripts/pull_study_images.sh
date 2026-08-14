#!/usr/bin/env bash
# Pull ONLY the images referenced by the human-study sample from rolf.
# Usage: scripts/pull_study_images.sh [run]   (default: main2b)
# Reads results/<run>/images_needed.txt (produced by make_human_study.py) and
# rsyncs exactly those files, then regenerates the HTML so they embed.
set -euo pipefail
RUN="${1:-main2b}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${ROLF_HOST:-rolf}"
RDIR="FailureModeBench/results/${RUN}/predictions/images"
LIST="${HERE}/results/${RUN}/images_needed.txt"
DEST="${HERE}/results/${RUN}/images"
[ -f "$LIST" ] || { echo "missing $LIST -- run make_human_study.py first"; exit 1; }
mkdir -p "$DEST"
echo "pulling $(wc -l < "$LIST") images from ${REMOTE}:${RDIR}/ ..."
rsync -az --files-from="$LIST" "${REMOTE}:${RDIR}/" "$DEST/"
echo "have $(ls "$DEST" | wc -l) images locally; regenerating HTML ..."
python3 "${HERE}/scripts/make_human_study.py" --run "$RUN" --n-per-family 20
