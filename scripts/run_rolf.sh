#!/bin/bash
# FailureModeBench — full pipeline on rolf (run inside `screen`).
#
#   CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh
#
# Env knobs:
#   RUN=main            results/<RUN>/
#   MODELS="qwen2vl_7b internvl25_8b llava16_7b"
#   LIMIT=              cap samples/dataset (smoke: LIMIT=20)
#   N_PER_FAMILY=200    errors judged per task family
#   PROVIDER=anthropic  judge backend (needs ANTHROPIC_API_KEY)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN=${RUN:-main}
MODELS=${MODELS:-"qwen2vl_7b internvl25_8b llava16_7b"}
MODELS_CSV=$(echo "$MODELS" | tr ' ' ',')     # cli takes comma-separated model keys
N_PER_FAMILY=${N_PER_FAMILY:-200}
PROVIDER=${PROVIDER:-anthropic}       # anthropic | openai | local (open VLM, no key)
JUDGE_MODEL=${JUDGE_MODEL:-}          # e.g. Qwen2.5-VL-7B-Instruct for PROVIDER=local
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
JUDGE_MODEL_ARG=""; [ -n "$JUDGE_MODEL" ] && JUDGE_MODEL_ARG="--judge-model $JUDGE_MODEL"
EMBED_ARG=""; [ -n "$EMBED" ] && EMBED_ARG="--embeddings"   # semantic label matching

echo "############ FailureModeBench run=$RUN models=[$MODELS] ############"

echo "===== STAGE 1/4: recognition inference (GPU) ====="
python -m failuremodebench.cli --run "$RUN" --models $MODELS_CSV $LIMIT_ARG $EMBED_ARG \
    infer-recognition

echo "===== STAGE 2/4: VQA inference via VLMEvalKit (GPU) ====="
python -m failuremodebench.cli --run "$RUN" --models $MODELS_CSV $LIMIT_ARG \
    infer-vqa

echo "===== STAGE 3/4: extract errors + failure-mode judge ($PROVIDER) ====="
python -m failuremodebench.cli --run "$RUN" --provider "$PROVIDER" $JUDGE_MODEL_ARG \
    --n-per-family "$N_PER_FAMILY" judge

echo "===== STAGE 4/4: aggregate tables + figures ====="
python -m failuremodebench.cli --run "$RUN" aggregate

echo
echo "DONE. Results under results/$RUN/ :"
echo "  tables/failure_by_family.csv   <- headline result"
echo "  tables/accuracy.csv, tables/confusion/*"
echo "  figures/*.png , summary.json"
echo "Pull back to laptop:  rsync -av rolf:~/FailureModeBench/results/$RUN/ results/$RUN/"
