# FailureModeBench

**A Systematic Failure-Mode Benchmark for Vision-Language Models — From Fine-Grained Recognition to Multimodal Reasoning.**

VLMs report strong aggregate accuracy but fail in ways aggregate scores hide.
FailureModeBench evaluates VLMs across **16 datasets** (8 recognition, 8 VQA),
then routes every error into an **8-category failure taxonomy (F1–F8)** with an
LLM judge, producing per-family failure-rate tables, confusion matrices and
figures.

```
recognition (HF classify) ┐
                          ├─► unified PredictionRecords ─► error corpus ─► LLM judge (F1–F8) ─► tables + figures
VQA (VLMEvalKit)          ┘
```

## The 8 failure modes (paper §3.2)

| Code | Mode | One-line |
|------|------|----------|
| F1 | Visual Hallucination | reports visual entities absent from the image |
| F2 | Linguistic Hallucination | unsupported verbal claims not grounded in input |
| F3 | Cross-Modal Misalignment | ignores the image / mis-binds text to it |
| F4 | Fusion Error | reads parts right, combines them wrongly |
| F5 | Fabrication | invents whole steps / facts / causal chains |
| F6 | Misattribution | right property, wrong entity or region |
| F7 | Overgeneralization | super-category instead of fine-grained subclass |
| F8 | Confabulation | fluent, coherent, but false reasoning chain |

## The 16 datasets (paper §3.1)

Recognition: `imagenet, imagenet_v2, imagenet_a, imagenet_r, imagenet_sketch,
dtd, food101, resisc45`
VQA: `mathverse, mathvista, seed, mme, realworldqa, capture, crpe, ai2d`

See `failuremodebench/config.py` for families, HF paths, VLMEvalKit ids and the
used/original test sizes.

## Layout

```
failuremodebench/
  config.py        registries: 16 datasets, models, task families
  taxonomy.py      F1–F8 definitions + LLM-judge rubric
  recognition.py   HF classification runner (ImageNet family, DTD, Food101, Resisc45)
  labelmatch.py    free-text answer -> canonical label
  backends.py      VLM backends: VLMEvalKit (GPU) / API (CPU smoke)
  vqa.py           VLMEvalKit driver + prediction importer
  records.py       unified PredictionRecord schema + JSONL IO
  judge.py         error extraction + failure-mode LLM judge (resumable)
  aggregate.py     failure-rate tables + confusion matrices
  figures.py       stacked failure bar + accuracy heatmap
  cli.py           `python -m failuremodebench.cli ...`
  labelsets/       generated class-name lists (build_labelsets.py)
scripts/           build_labelsets, sync_to_rolf, bootstrap_rolf, run_rolf
configs/matrix.yaml  the evaluation matrix
tests/test_pipeline.py  offline end-to-end validation (no GPU/key)
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/build_labelsets.py          # one-time (network)
python tests/test_pipeline.py              # offline self-test (no GPU/key)
```

Full evaluation runs on a GPU box — see **docs/LAUNCH.md**. On rolf:

```bash
bash scripts/sync_to_rolf.sh               # from laptop
# on rolf:
bash scripts/bootstrap_rolf.sh
export ANTHROPIC_API_KEY=sk-ant-...
screen -S fmb
CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh
```

## Pipeline commands

```bash
python -m failuremodebench.cli --run main infer-recognition   # GPU
python -m failuremodebench.cli --run main infer-vqa           # GPU (VLMEvalKit)
python -m failuremodebench.cli --run main judge               # LLM judge F1–F8
python -m failuremodebench.cli --run main aggregate           # tables + figures
```

Outputs land in `results/<run>/`: `predictions/*.jsonl`, `corpus.jsonl`,
`tables/failure_by_family.csv` (headline), `tables/confusion/*`,
`figures/*.png`, `summary.json`.

## Status

Harness complete and validated offline (recognition data path on real DTD;
judge→aggregate→figures on synthetic corpora). GPU inference is wired but not yet
executed — pending rolf reachability + a judge API key (see docs/LAUNCH.md).
