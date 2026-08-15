# FailureModeBench

**A Systematic Failure-Mode Benchmark for Vision-Language Models — From Fine-Grained Recognition to Multimodal Reasoning.**

*Target venue: AAAI-27 Main Technical Track (Multimodal / Computer Vision / Evaluation).*

---

## 1. What this is

Vision-Language Models (VLMs) report strong **aggregate** accuracy, yet fail in
ways aggregate scores hide: a model can score 85% on ImageNet and still
systematically confuse fine-grained textures, or invent mathematical steps on
MathVerse. Existing suites report a single number per dataset; they rarely
diagnose **how** and **why** a model fails.

FailureModeBench is a diagnostic benchmark that:

1. evaluates VLMs across **16 datasets** (8 recognition, 8 VQA) spanning generic
   robustness, fine-grained recognition, remote sensing, mathematical reasoning,
   general understanding, real-world QA, counterfactual understanding,
   compositionality/hallucination, and chart understanding;
2. routes **every error** into an **8-category failure taxonomy (F1–F8)** using
   an LLM judge; and
3. reports **per-family failure-rate tables**, **confusion matrices** for
   fine-grained classes, and figures — a standardized, reproducible protocol.

```
 recognition (HF classify) ┐
                           ├─►  unified PredictionRecords  ─►  error corpus  ─►  LLM judge (F1–F8)  ─►  tables + figures
 VQA (VLMEvalKit)          ┘
```

The core idea: **unify recognition and reasoning errors under one failure
taxonomy**, so you can say things like *"fine-grained recognition is dominated by
Overgeneralization and Cross-Modal Misalignment, while VQA failures cluster
around Fabrication and Confabulation."*

---

## Reproduce (no GPU / no API key)

All model predictions and LLM-judge labels are cached under `results/*/`. Regenerate
every table, figure, and the judge-validity analysis from that committed data:

```bash
./reproduce.sh                 # installs deps + runs everything  (shell entry point)
# or, equivalently:
python reproduce.py            # tables + figures + judge-validity (Python entry point)

# individual stages (either entry point):
./reproduce.sh figures         # -> docs/figures/fig1-7.pdf + PNGs
./reproduce.sh agreement       # human-vs-judge + Fleiss' kappa (reads human_annotations/)
./reproduce.sh tables          # per-family F1-F8 tables -> results/main2b/tables/
./reproduce.sh --no-install agreement   # skip the pip install step
```

**Re-running the heavy stages** (model inference + LLM judging; needs a GPU and an
OpenAI-compatible judge key) uses the CLI:

```bash
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1   # judge
python -m failuremodebench.cli --run RUN --models M --datasets D infer-recognition  # / infer-vqa / infer-clip
python -m failuremodebench.cli --run RUN judge          # LLM-judge errors -> corpus.jsonl
python -m failuremodebench.cli --run RUN aggregate      # tables + summary
```

## Repository layout

```
failuremodebench/     core package (config, taxonomy, backends, recognition, vqa,
                      judge, multijudge, aggregate, providers, cli)
scripts/              make_figures.py, multi_annotator_agreement.py, make_human_study.py,
                      expand_corpus.py, extract_vqa_images.py, ... (analysis + data prep)
human_annotations/    annotator1_ahmed.json, annotator2_cui.json  (+ _archive/, README)
results/<run>/        predictions/, corpus.jsonl (judged errors), tables/, figures/
docs/                 RESULTS_SUMMARY.md, SETUP_AND_RESULTS.md,                       ABSTRACT_INTRO.md, FIGURE_CAPTIONS.md, figures/*.pdf, annotation study
reproduce.py          single entry point (analysis from committed data)
```

Key result docs: **`docs/RESULTS_SUMMARY.md`** (paper-ready), `docs/SETUP_AND_RESULTS.md`
(detailed methods + per-dataset tables), `docs/FIGURE_CAPTIONS.md`.

---

## 2. The 8-category failure taxonomy (§3.2)

Every incorrect prediction is assigned **exactly one** mode by the judge.

| Code | Failure mode | Definition | Typical manifestation |
|------|--------------|------------|-----------------------|
| **F1** | Visual Hallucination | Reports visual entities/features **absent** from the image | ImageNet-A: "sees" patterns that aren't there; RealWorldQA: non-existent objects |
| **F2** | Linguistic Hallucination | Unsupported verbal claims not grounded in image or question | Invented attributes; CRPE: details not in the caption |
| **F3** | Cross-Modal Misalignment | Ignores the image / mis-binds text to it | Food101: wrong ingredient names despite correct image; DTD: mismatched texture label |
| **F4** | Fusion Error | Reads the parts right, **combines** them wrongly | AI2D: misreads a chart axis; MathVista: combines image numbers incorrectly |
| **F5** | Fabrication | Invents whole facts / steps / relationships | MathVerse: invents a formula step; Capture: invents a causal chain |
| **F6** | Misattribution | Right property/action, **wrong entity or region** | ImageNet-R: texture to wrong class; SEED: action to wrong object |
| **F7** | Overgeneralization | Super-category instead of the required fine-grained subclass | ImageNet: "bird" not "sparrow"; Resisc45: "forest" not the specific land-use |
| **F8** | Confabulation | Fluent, coherent, but **false** reasoning chain | MME: fluent-but-wrong explanation; CRPE: inverted compositional relation |

Definitions and the judge rubric live in
[`failuremodebench/taxonomy.py`](failuremodebench/taxonomy.py) — the single source
of truth shared by the prompt and the parser.

---

## 3. The 16 datasets (§3.1)

Two evaluation **modalities**:
- **recognition** — image classification. The VLM names the class; the free-text
  answer is resolved to a fixed label set. Scored with top-1 accuracy (+ macro-F1
  / confusion matrix for fine-grained sets).
- **vqa** — (MCQ/open) visual QA. Run through **VLMEvalKit**, which owns loaders,
  prompt templates and answer extraction.

| Family | Dataset | Modality | Focus | Used / Original |
|--------|---------|----------|-------|-----------------|
| Generic | ImageNet | recognition | Generic | 49,032 / 50,000 |
| Generic | ImageNet-V2 | recognition | Generic | 9,772 / 10,000 |
| Robustness | ImageNet-A | recognition | Generic (adv.) | 7,467 / 7,500 |
| Robustness | ImageNet-R | recognition | Texture | 28,506 / 30,000 |
| Robustness | ImageNet-Sketch | recognition | Edges | 35,350 / 50,000 |
| Robustness | DTD | recognition | Edges, Texture | 5,640 / 5,640 |
| Fine-grained | Food101 | recognition | Fine-grained | 25,250 / 25,250 |
| Satellite | Resisc45 | recognition | Satellite imagery | 4,500 / 4,500 |
| Math | MathVerse (MCQ) | vqa | Mathematical ability | 1,631 / 2,180 |
| Math | MathVista | vqa | Mathematical ability | 490 / 1,000 |
| General VQA | SEED | vqa | General understanding | 3,881 / 13,991 |
| General VQA | MME | vqa | General understanding | 1,576 / 2,370 |
| Real-World | RealWorldQA | vqa | Real-world understanding | 765 / 765 |
| Counterfactual | Capture | vqa | Counterfactual understanding | 817 / 962 |
| General VQA | CRPE | vqa | General (compositional relations & hallucination) | 7,575 / 7,575 |
| Chart | AI2D | vqa | Graph & chart understanding | 2,704 / 3,090 |

*Preprocessing:* images with any side > 1000 px are dropped; "Used" reflects
post-filter counts. Exact HF paths and VLMEvalKit ids are in
[`failuremodebench/config.py`](failuremodebench/config.py).

> **Two ids to verify on rolf** (VLMEvalKit ids drift across versions):
> `crpe` (pick the right CRPE split) and `capture` (currently a `COCO_VAL`
> **placeholder** — set to the actual counterfactual dataset id or drop it).
> See the **Reproduce** section above.

---

## 4. Models under test (§3.3)

Open weights (GPU, run on rolf) and a closed API option:

| Key | Model | Weights | 4-bit |
|-----|-------|---------|-------|
| `qwen2vl_7b` | Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` | yes |
| `internvl25_4b` | InternVL2.5-4B | `OpenGVLab/InternVL2_5-4B` | no |
| `internvl25_8b` | InternVL2.5-8B | `OpenGVLab/InternVL2_5-8B` | yes |
| `llava16_7b` | LLaVA-1.6-7B (vicuna) | `llava-hf/llava-v1.6-vicuna-7b-hf` | yes |
| `llava16_13b` | LLaVA-1.6-13B (vicuna) | `llava-hf/llava-v1.6-vicuna-13b-hf` | yes |
| `claude_opus` | Claude (closed) | API | — |

Default matrix: `qwen2vl_7b, internvl25_8b, llava16_7b`. The 4B/8B and 7B/13B
pairs are there for size-variant ablations.

> **11 GB note:** rolf's RTX 2080 Ti has 11 GB, so 7B/8B/13B load in 4-bit. For
> Qwen2-VL (dynamic resolution) cap the max vision tokens to avoid OOM.

---

## 5. Repository layout

```
failuremodebench/
  config.py        registries: 16 datasets, models, task families, resolution filter
  taxonomy.py      F1–F8 definitions + LLM-judge system/user prompts + JSON schema
  recognition.py   HF classification runner (ImageNet family, DTD, Food101, Resisc45)
  labelmatch.py    free-text answer -> canonical label (substring/synonym/Jaccard/embed)
  backends.py      VLM backends: VLMEvalKit (GPU) or API (CPU smoke test)
  vqa.py           VLMEvalKit driver + prediction-spreadsheet importer
  records.py       unified PredictionRecord schema + JSONL IO + image saving
  providers.py     LLM providers for the judge/API-VLM (Anthropic / OpenAI / echo stub)
  judge.py         error extraction + stratified sampling + failure-mode judge (resumable)
  aggregate.py     failure-rate tables + confusion matrices
  figures.py       stacked failure bar + accuracy heatmap (Agg backend, no X server)
  cli.py           `python -m failuremodebench.cli ...`
  labelsets/       generated class-name lists (imagenet, food101, dtd, resisc45, ...)
  discovery.py     OPTIONAL/experimental (unwired) — automated failure-slice discovery
scripts/
  build_labelsets.py   one-time: generate labelsets/*.json from HF metadata
  sync_to_rolf.sh      laptop -> rolf (code + labelsets, single OTP via ssh multiplexing)
  bootstrap_rolf.sh    on rolf: conda env + VLMEvalKit + sanity checks
  run_rolf.sh          on rolf: the full 4-stage pipeline
configs/matrix.yaml    the evaluation matrix (models x datasets x judge settings)
tests/test_pipeline.py offline end-to-end validation (no GPU, no key)
results/<run>/         all outputs (git-ignored)
```

---

## 6. How the pipeline works

Unified record schema ([`records.py`](failuremodebench/records.py)) — **both**
runners emit the same JSON, so everything downstream is modality-blind:

```json
{"sample_id","dataset","family","modality","model",
 "question","gold","prediction","pred_label","correct","image_ref","extra"}
```

**Stage 1 — recognition inference.** Stream each HF split, apply the >1000 px
filter, prompt the VLM to name the class, resolve the answer to a label
(exact → synonym → token-Jaccard → optional semantic embedding), score top-1.
Only *errors* keep a saved image (bounds disk).

**Stage 2 — VQA inference.** Shell out to VLMEvalKit per (model, dataset); import
its per-sample prediction spreadsheet (tolerant to column-name drift) into the
unified schema, carrying over VLMEvalKit's own correctness column.

**Stage 3 — failure-mode judge.** Gather all errors; per family, take a
deterministic stratified sample of up to `N_PER_FAMILY` (default 200), balanced
across (model, dataset); ask the judge (Claude) to assign one F1–F8 with a
rationale + confidence, attaching the image. **Resumable**: the corpus is written
incrementally and de-duplicated by content hash, so a rate-limited/crashed run
picks up where it stopped.

**Stage 4 — aggregate.** Emit `accuracy.csv`, `failure_by_{family,dataset,model}.csv`,
`confusion/<model>__<ds>.csv` for fine-grained sets, plus figures and a
`summary.json` (including the dominant mode per family — the headline claim).

---

## 7. Install

```bash
pip install -r requirements.txt          # core deps
python scripts/build_labelsets.py        # one-time; needs network to HuggingFace
python tests/test_pipeline.py            # offline self-test (no GPU, no key)
```

Expected self-test tail:
```
[ok] gathered … errors, sampled … across … families
[ok] echo judge wrote … verdicts, all parsed
[ok] judge resume is idempotent
[ok] aggregation tables written; dominant/family = {...}
[ok] confusion matrices: [...]
[ok] figures: ['failure_by_family.png', 'accuracy_heatmap.png']
ALL PIPELINE CHECKS PASSED
```

`vlmeval` (VQA half) and torch/CUDA are installed on rolf by
`bootstrap_rolf.sh`, not needed for the laptop self-test.

---

## 8. Run the full evaluation (on the rolf GPU box)

You do **not** download datasets/models by hand — the run pulls them from
HuggingFace on rolf, cached to `/local/scratch` (not your home quota).

```bash
# --- from your laptop (must be on the UZH campus network / VPN) -------------
bash scripts/sync_to_rolf.sh                     # pushes code+labelsets; one OTP

# --- on rolf ----------------------------------------------------------------
cd ~/FailureModeBench
bash scripts/bootstrap_rolf.sh                   # conda env 'fmb' + VLMEvalKit
huggingface-cli login                            # once; imagenet-1k is gated
export ANTHROPIC_API_KEY=sk-ant-...              # judge key (or --provider openai)

screen -S fmb                                    # survive disconnects
conda activate fmb

# shake-out first (20 samples/dataset — catches ids/memory in minutes):
RUN=smoke LIMIT=20 N_PER_FAMILY=20 CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh

# full run:
RUN=main CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh 2>&1 | tee run_rolf.log

# --- pull results back to laptop --------------------------------------------
rsync -av rolf:~/FailureModeBench/results/main/ results/main/
```

Run individual stages with the CLI:
```bash
python -m failuremodebench.cli --run main infer-recognition
python -m failuremodebench.cli --run main infer-vqa
python -m failuremodebench.cli --run main --n-per-family 200 judge
python -m failuremodebench.cli --run main aggregate
```

Useful flags: `--models qwen2vl_7b llava16_7b`, `--datasets dtd food101 ai2d`,
`--limit 50`, `--provider openai`, `--embeddings` (semantic label matching),
`--echo` (offline stub judge), `--no-images` (text-only judge).

---

## 9. Outputs (`results/<run>/`)

```
predictions/<model>__<dataset>.jsonl   per-sample unified records
images/                                saved images for judged errors
corpus.jsonl                           annotated error corpus (F1–F8 verdicts)
tables/
  accuracy.csv                         top-1 accuracy / VQA score per (model, dataset)
  failure_by_family.csv                F1–F8 distribution (%) per family   ← HEADLINE
  failure_by_dataset.csv               … per dataset
  failure_by_model.csv                 … per model
  confusion/<model>__<ds>.csv          per-class confusion (fine-grained sets)
figures/
  failure_by_family.png                stacked bar of F1–F8 by family
  accuracy_heatmap.png                 model × dataset accuracy
summary.json                           counts + dominant failure mode per family
```

`failure_by_family.csv` columns: `family, n_errors, F1..F8 (counts), F1_pct..F8_pct`.
Percentages per row sum to ~100 — drop straight into a LaTeX table.

---

## 10. Configuration & extending

- **Change the matrix:** edit [`configs/matrix.yaml`](configs/matrix.yaml) or pass
  `--models` / `--datasets` on the CLI. `run_rolf.sh` reads `RUN`, `MODELS`,
  `LIMIT`, `N_PER_FAMILY`, `PROVIDER` env knobs.
- **Add a dataset:** add a `DatasetSpec` to `DATASETS` in `config.py`
  (recognition: `hf_path` + a `labelsets/*.json`; vqa: `vlmevalkit_name`).
- **Add a model:** add a `ModelSpec` to `MODELS` (`vlmevalkit_name` for the GPU
  path, or `family="api"` for a closed model).
- **Add/rename a failure mode:** edit `FAILURE_MODES` in `taxonomy.py`; the rubric
  and JSON schema update automatically.
- **Swap the judge:** `--provider anthropic|openai`; keys via
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.
- **2-shot / chain-of-visual-thought:** extend the prompt in `recognition.py` /
  the VLMEvalKit call in `vqa.py` (hooks noted in `configs/matrix.yaml`).

---

## 11. Reproducibility notes

- Sampling for the judge is **deterministic** (no RNG) — same predictions →
  same annotated corpus.
- Judge runs at `temperature=0`.
- Label sets are generated from HF metadata and committed under
  `failuremodebench/labelsets/` so recognition scoring is stable.
- The resolution filter (1000 px) and used/original counts are asserted against
  the paper's Table.

---

## 12. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ssh: connect … port 22: timed out` | Not on campus — connect UZH VPN or use a `ProxyJump` |
| `imagenet-1k is a gated dataset` | `huggingface-cli login` on rolf (one time) |
| CUDA OOM on a 7B/13B model | ensure `load_4bit`; cap Qwen2-VL max pixels; one model per GPU |
| `… not in VLMEvalKit supported_VLM` | version mismatch — check the id: `python -c "from vlmeval.config import supported_VLM; print(list(supported_VLM))"` |
| No prediction file imported for a VQA dataset | verify the `vlmevalkit_name` (esp. `crpe`, `capture`) exists in your VLMEvalKit version |
| `ANTHROPIC_API_KEY is not set` | export it before `judge`, or use `--echo` for a keyless plumbing test |
| Judge output unparseable | it's retried 3× then recorded as `NA`; check the `rationale` field in `corpus.jsonl` |

---

## 13. Status

Harness **complete and validated offline**: recognition data path on real DTD;
`judge → aggregate → figures` on synthetic corpora; all modules compile and the
self-test passes. GPU inference is wired but not yet executed — pending a rolf
launch (campus reach + `huggingface-cli login` + `ANTHROPIC_API_KEY`). See
the **Reproduce** section above.

## 14. Citation

```bibtex
@inproceedings{failuremodebench2027,
  title     = {FailureModeBench: A Systematic Failure-Mode Benchmark for
               Vision-Language Models --- From Fine-Grained Recognition to
               Multimodal Reasoning},
  author    = {TODO},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027}
}
```
