# Planned robustness runs (reviewer W2/W4) — paste-ready

Run on rolf when a GPU is free; both are small (~20 min each). Then re-judge the new
errors and compare F7 to the main closed-set result.

## R1 — Ordering-bias check (W2/Q3): F7 invariant to label order
```bash
source /local/scratch/alali/miniconda/etc/profile.d/conda.sh; conda activate fmb
export HF_HOME=/local/scratch/alali/hf_cache CUDA_VISIBLE_DEVICES=<G> \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for S in 1 2 3; do
  python -m failuremodebench.cli --run closedset_shuf$S --models qwen2vl_2b \
    --datasets food101,dtd,resisc45 --limit 1000 --embeddings \
    --label-hint-k 110 --shuffle-labels $S infer-recognition
done
# expectation: accuracy + F7 stable across seeds (free-form answer, no positions)
```

## R2 — Large-label closed-set (W4/Q5): does F7 collapse persist as candidates grow?
```bash
export CUDA_VISIBLE_DEVICES=<G>
for K in 9 49 199; do
  python -m failuremodebench.cli --run closedset_in_k$K --models qwen2vl_2b \
    --datasets imagenet --limit 1000 --embeddings \
    --label-hint-k 100000 --hint-distractors $K infer-recognition
done
# expectation: F7 rises gradually with K but stays well below the open-ended 76-95%,
# bounded below by the contrastive baseline (~22%)
```

## Then (no GPU): judge the new errors + compare
```bash
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1
for R in closedset_shuf1 closedset_shuf2 closedset_shuf3 closedset_in_k9 closedset_in_k49 closedset_in_k199; do
  touch results/$R/corpus.jsonl
  python scripts/expand_corpus.py --run $R --n 200 --judge
done
```
