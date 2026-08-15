#!/usr/bin/env bash
# FailureModeBench — shell entry point.
# Regenerates all analysis artifacts (tables, figures, judge-validity) from the
# committed corpora in results/*. No GPU / no API key required.
#
#   ./reproduce.sh              # install deps (if needed) + run everything
#   ./reproduce.sh figures      # just the figures
#   ./reproduce.sh agreement    # just the judge-validity numbers
#   ./reproduce.sh tables       # aggregate the per-family tables
#   ./reproduce.sh --no-install <stage>   # skip the pip install step
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

INSTALL=1
if [[ "${1:-}" == "--no-install" ]]; then INSTALL=0; shift; fi
STAGE="${1:-all}"

if [[ "$INSTALL" == "1" ]]; then
  echo "[reproduce] installing requirements ..."
  "$PY" -m pip install -q -r requirements.txt
fi

echo "[reproduce] stage: $STAGE"
"$PY" reproduce.py "$STAGE"

cat <<'EOF'

[reproduce] done.
  figures        -> docs/figures/fig1-7.pdf + docs/FailureModeBench_figures.pdf
  tables         -> results/main2b/tables/
  judge-validity -> printed above (per-annotator vs judge, Fleiss' kappa, consensus)

To re-run the heavy stages (model inference + LLM judging; needs a GPU + judge key):
  export OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1
  python -m failuremodebench.cli --run RUN --models M --datasets D infer-recognition
  python -m failuremodebench.cli --run RUN judge
  python -m failuremodebench.cli --run RUN aggregate
EOF
