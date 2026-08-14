# FailureModeBench — Experimental Setup & Results (detailed)

All numbers are computed from the released per-error corpora (`results/*/corpus.jsonl`)
and predictions (`results/*/predictions/`). Primary model: Qwen2-VL-2B; recognition
accuracy on full test data; 20,147 generative errors judged in full.

---

## 1. Experimental Setup

### 1.1 Models under test (7 models, two paradigms)

**Generative VLMs** (free-form text output), four models across three architecture
families and sizes 2B–8B:

| Model | HF id | Arch | Size | Quantization |
|---|---|---|---|---|
| Qwen2-VL-2B-Instruct | `Qwen/Qwen2-VL-2B-Instruct` | Qwen2-VL | 2B | 4-bit nf4 (bf16 compute) |
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` | Qwen2-VL | 7B | 4-bit nf4 |
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | Qwen2.5-VL | 3B | 4-bit nf4 |
| InternVL2.5-8B | `OpenGVLab/InternVL2_5-8B` | InternVL | 8B | 4-bit nf4 |
| LLaVA-1.6-Vicuna-7B | `llava-hf/llava-v1.6-vicuna-7b-hf` | LLaVA-1.6 | 7B | 4-bit nf4 |

**Contrastive (CLIP-style) recognizers**, three newest-generation encoders evaluated
zero-shot via `open_clip`:

| Model | open_clip spec | Params |
|---|---|---|
| SigLIP2-SO400M/16 @384 | `ViT-SO400M-16-SigLIP2-384:webli` | ~0.9B |
| DFN-CLIP-H/14 @378 | `ViT-H-14-378-quickgelu:dfn5b` | ~0.6B |
| MetaCLIP2-H/14 | `ViT-H-14-quickgelu:metaclip_fullcc` | ~0.6B |

**Zero-shot classification protocol (CLIP):** each image is embedded once; each class
label is embedded with a mean-ensemble of two prompt templates ("a photo of a {}.",
"a photo of the {}."); the predicted class is the arg-max cosine similarity. The model
is therefore *constrained to the label set* — it cannot emit an out-of-set token — which
is central to the validity control (§5).

**Infrastructure & reproducibility notes.** All models run 4-bit-quantized on a single
shared 11 GB RTX 2080 Ti (contended cluster). Three engineering fixes were required and
are released with the harness: (i) a **processor-level `max_pixels` cap** (96·28·28) so
Qwen2-VL's dynamic-resolution vision tower fits 11 GB on large-image datasets (MathVista,
NaturalBench) — without it these OOM at ~12 GB in a single attention allocation; (ii) a
**non-streaming HuggingFace loader** to avoid a streaming deadlock; (iii) a separate
`transformers==4.46` environment for InternVL2.5 and LLaVA-1.6, whose VLMEvalKit wrappers
break under the `transformers 5.14` required by Qwen2.5-VL (sentencepiece, tied-weights,
`GenerationMixin`, and KV-cache API changes). Qwen models use `transformers 5.14`.

### 1.2 Tasks and datasets (9 families, 16 datasets)

| Modality | Family | Datasets | Test size (approx.) |
|---|---|---|---|
| Recognition | fine-grained | Food101 | 25,250 |
| Recognition | generic | ImageNet, ImageNet-V2 | 49k / 10k |
| Recognition | robustness | ImageNet-A, -R, -Sketch, DTD | 7.5k / 30k / 50k / 5.6k |
| Recognition | satellite | RESISC45 | 4.5k |
| VQA | chart/diagram | AI2D | 3,088 |
| VQA | general | SEED, MME, CRPE | 24k / 2.4k / 3.9k |
| VQA | real-world | RealWorldQA | 765 |
| VQA | math | MathVista, MathVerse | 1k / 3.9k |
| VQA | counterfactual | NaturalBench | 7.6k |

Recognition accuracy is on **full** test data (CLIP: all 165,430 images/model). VQA uses
the full VLMEvalKit sets. Generative recognition resolves free-text answers to the label
set with a four-stage matching cascade: exact substring → curated synonyms → Jaccard
token overlap → sentence-embedding nearest label (CLIP-benchmark standard); accuracy is
reported under the embedding stage and we ablate the cascade (matching-sensitivity is
itself the F7 signature — exact-match ≈0%).

### 1.3 The failure taxonomy and judge

Every *incorrect* prediction is routed to exactly one of eight modes:

| Code | Mode | One-line meaning | Example (Qwen2-VL-2B) |
|---|---|---|---|
| F1 | Visual Hallucination | reports a visual entity not in the image | mme: gold "No" → "Yes" |
| F2 | Linguistic Hallucination | unsupported verbal claim | mme: gold "No" → "yes" |
| F3 | Cross-Modal Misalignment | ignores/mis-binds the image | realworldqa: gold "B" → "A. 0" |
| F4 | Fusion Error | facts read right, combined wrong | mathvista / realworldqa |
| F5 | Fabrication | invents whole steps/relations | ai2d: gold "B" → "C. filtering" |
| F6 | Misattribution | right property, **wrong entity** | food101: **beignets → "donuts"** |
| F7 | Overgeneralization | super-category, not the subclass | food101: **beignets → "breakfast"** |
| F8 | Confabulation | fluent but false reasoning | mme: gold "Yes" → "No" |

The **F6 vs F7 pair is the crux**: *beignets→"donuts"* (F6, a look-alike sibling) vs
*beignets→"breakfast"* (F7, the parent category). The judge applies a fixed priority
order (F7 first if a broader parent is named; then F1/F2 for absent entities; F4 for
mis-combination; F5 for invention; F6 for wrong-entity; F8 for fluent-but-false; F3 for
ignoring the image) and returns strict JSON `{failure_mode, confidence, rationale,
secondary_mode}`.

**Primary judge:** `deepseek-v4-flash` via OpenRouter, **text-only** (GPU-free), 2048
output tokens. It is a *reasoning* model — a critical detail: at a 512-token cap the
hidden reasoning pass consumes the budget before any JSON is emitted, silently producing
"NA" on the hardest 12.5% of errors; 2048 tokens fixes this (0 NA). Judging is issued
concurrently; **20,147 generative errors were judged in full** (not sampled); contrastive,
per-architecture, and closed-set errors were judged on large samples (600–1,000/model).

### 1.4 Evaluation protocol

Accuracy = full-data, cascade-resolved (recognition) or VLMEvalKit-scored (VQA). Failure
distribution = one LLM-judge label per error. Judge validity is established three
independent ways (§2). Recognition families for the F7 analysis are the four recognition
families (fine-grained, generic, robustness, satellite).

---

## 2. LLM-Judge Reliability

### 2.1 Inter-judge agreement (8-judge cross-provider panel)

Panel: deepseek-v4-flash, gpt-4o, gpt-4o-mini, qwen-2.5-72b, llama-3.3-70b, mistral-large,
deepseek-chat, phi-4 (all via OpenRouter, text-only, 2048 tokens).

- **Mean pairwise Cohen's κ = 0.52–0.56**, range **0.32–0.76** over 28 judge pairs.
- **All eight judges independently rank F7 as the dominant recognition mode.** The
  headline is robust to judge choice; disagreement concentrates on secondary modes
  (notably F3 vs F6, and F1 vs F6 on VQA).

### 2.2 Which judge best matches humans (455 human-labeled errors)

| Judge | raw agreement | κ vs human |
|---|---|---|
| **gpt-4o** | 70.9% | **0.58** |
| deepseek-v4-flash (deployed) | 63.1% | 0.47 |
| mistral-large | 57.6% | 0.41 |
| deepseek-chat | 56.1% | 0.40 |
| phi-4 | 52.2% | 0.37 |
| llama-3.3-70b | 49.3% | 0.33 |
| gpt-4o-mini | 46.1% | 0.30 |
| majority-vote ensemble | 61.5% | 0.46 |

(i) gpt-4o is the most human-aligned judge; the deployed GPU-free deepseek judge is a
solid second and far ahead of the smaller models — model *tier* does not predict human
alignment (gpt-4o-mini and llama-70b are lowest). (ii) The **ensemble underperforms the
best single judge** (0.46 < 0.58): pooling a strong judge with weaker ones dilutes rather
than denoises. We therefore report a single judge, not an ensemble.

### 2.3 Human study (two independent annotators, blind)

Two expert annotators independently labeled the study errors blind to the judge's
verdict. **The judge sits inside the human inter-annotator envelope** — the strongest
available evidence that the automated taxonomy is valid.

**Pairwise Cohen's κ (407 items rated by both humans and the judge):**

| | Annotator 1 | Annotator 2 | LLM judge |
|---|---|---|---|
| Annotator 1 | — | 0.471 | 0.428 |
| Annotator 2 | 0.471 | — | **0.552** |
| LLM judge | 0.428 | 0.552 | — |

Human–human agreement is κ **0.471**; the judge's agreement with the two humans (0.428,
0.552) **brackets** it — i.e. the judge disagrees with a human no more than two humans
disagree with each other. Adding the judge to the two humans barely changes **Fleiss' κ
(0.462 → 0.480)**: the judge behaves statistically like a third annotator.

**Judge vs human consensus.** On the 249 items where both humans agree, the judge matches
**85.1%**, and **99–100% on fine-grained/generic/robustness** — the families carrying the
headline. Disagreement is specific and localised:

| Family (both humans agree) | judge matches |
|---|---|
| fine-grained | 100% |
| robustness | 100% |
| generic | 99% |
| chart | 82% |
| general-VQA | 64% |
| real-world | 25% |
| satellite | 6% |

The satellite outlier is a genuine, reportable disagreement (both humans read coarse
land-use errors differently from the judge), not noise. Individually, annotator-vs-judge
is κ 0.464 (raw 62.6%, n=591) and 0.422 (raw 60.0%, n=495). **Dataset-mislabel rate**
(annotator flags gold-wrong / model-correct): 1.1% and 8.6% for the two annotators — the
annotators differ in how liberally they flag noise, itself a note on gold quality.
(Fig. `judge_validity.png` / `fig8_judge_validity.pdf`.)

### 2.4 A disclosed evaluation-validity lesson

The blind study surfaced that a **reasoning-model judge under a tight token budget**
silently mislabels — the 512→2048 fix recovered 12.5% of errors from "NA". We report this
as a reusable caution for automated-evaluation pipelines.

---

## 3. Failure Modes: 3 Generative Architectures vs 3 Contrastive Models

### 3.1 Per-dataset recognition accuracy (all models)

| Dataset | Qwen2-VL-2B | Qwen2-VL-7B | Qwen2.5-VL-3B | InternVL2.5-8B | LLaVA-1.6-7B | SigLIP2 |
|---|---|---|---|---|---|---|
| Food101 | 4.9 | 47.7 | 27.3 | 29.2 | 11.5 | **96.5** |
| ImageNet | 14.1 | 25.2 | 21.3 | 21.6 | 19.1 | **82.3** |
| ImageNet-V2 | 14.5 | 38.8 | 26.3 | 22.7 | 14.6 | **77.2** |
| DTD | 7.6 | 13.9 | 10.6 | 8.3 | 11.1 | **67.9** |
| ImageNet-A | 1.8 | 18.1 | 4.6 | 8.6 | 4.3 | **85.2** |
| ImageNet-R | 10.4 | 35.8 | 21.0 | 13.2 | 15.6 | **95.3** |
| ImageNet-Sketch | 15.5 | 38.0 | 28.1 | 23.1 | 21.3 | **73.6** |
| RESISC45 | 50.4 | 50.7 | 50.4 | 43.7 | 63.1 | **68.4** |

Contrastive models dominate on every recognition dataset — most extremely on
fine-grained (Food101 **96.5% vs 4.9–29%**) — and the gap is widest exactly where F7 is
highest.

### 3.2 Recognition failure-mode distribution by model (% of recognition errors)

| Model | acc | F1 | F3 | F6 | **F7** |
|---|---|---|---|---|---|
| Qwen2-VL-2B | 14.9% | 11 | 2 | 8 | **76** |
| Qwen2-VL-7B | 33.5% | 3 | 0 | 8 | **66** |
| Qwen2.5-VL-3B | 23.7% | 12 | 0 | 13 | **73** |
| InternVL2.5-8B | 21.3% | 8 | 2 | 14 | **75** |
| LLaVA-1.6-7B | 20.1% | 8 | 0 | 8 | **83** |
| **SigLIP2 / DFN / MetaCLIP2** | **82.1%** | 10 | 11 | **56** | **22** |

**Every generative model is F7-dominant (66–83%)** across three architectures and sizes
2B–8B. **All three contrastive models are F6-dominant (56%)** with F7 collapsed to 22%.
Overgeneralization is a property of the generative paradigm, not of any one model.
(Figs. `failuremode_by_model.png`, `f7_by_architecture.png`, `acc_vs_f7_scatter.png`.)

### 3.3 Scale (same architecture, Qwen2-VL 2B→7B)

Accuracy roughly doubles (14.9→33.5% overall; Food101 **4.9→47.7%**, ImageNet-A
**1.8→18.1%**) and F7 falls 76→66%, while F1 hallucination drops 11→3%. Scale reduces
overgeneralization and improves granularity — but **F7 remains the dominant mode** at 7B.
(Fig. `scale_qwen2vl.png`.)

---

## 4. Dominating Failure Modes Across Tasks (9 families, per-dataset)

Full-data judging of all 20,147 Qwen2-VL-2B errors, by dataset:

| Family | Dataset | n | Dominant | Dominant % | F7 % |
|---|---|---|---|---|---|
| fine-grained | Food101 | 951 | **F7** | 99 | 99 |
| generic | ImageNet | 859 | **F7** | 95 | 95 |
| generic | ImageNet-V2 | 855 | **F7** | 81 | 81 |
| robustness | ImageNet-A | 982 | **F7** | 87 | 87 |
| robustness | ImageNet-Sketch | 845 | **F7** | 84 | 84 |
| robustness | ImageNet-R | 896 | **F7** | 54 | 54 |
| robustness | DTD | 924 | **F7** | 38 | 38 |
| satellite | RESISC45 | 496 | **F7** | 67 | 67 |
| chart | AI2D | 778 | **F6** | 72 | 2 |
| general | CRPE | 1,665 | **F6** | 77 | 1 |
| general | SEED | 3,860 | **F6** | 47 | 3 |
| general | MME | 480 | **F6** | 47 | 1 |
| math | MathVerse | 2,901 | **F4** | 78 | 0 |
| math | MathVista | 762 | **F4** | 59 | 2 |
| real-world | RealWorldQA | 287 | **F1** | 40 | 0 |
| counterfactual | NaturalBench | 2,606 | **F1** | 45 | 1 |

**Four regions under identical weights:**
1. **Recognition → F7** (all 8 recognition datasets; 38–99%). Right object, wrong
   granularity. The gradient (Food101 99% → DTD 38%) tracks how fine-grained the label
   space is.
2. **Structured VQA → F6** (AI2D, CRPE, SEED, MME; 47–77%). Right property, wrong entity.
3. **Math → F4** (MathVista, MathVerse; 59–78%). Right quantities, wrong combination.
4. **Real-world / counterfactual → F1** (RealWorldQA, NaturalBench; 40–45%). Claims
   things not in the image.

Accuracy alone sees none of this structure — it collapses four qualitatively different
behaviors into one number. (Fig. `failure_by_family.png`, `accuracy_heatmap.png`.)

**Hypotheses adjudicated.** The taxonomy was built to test specific predictions: (i)
fine-grained recognition is F7-dominated — **confirmed** (99%); (ii) VQA moves away from
F7 toward F6/F1 — **confirmed** (F7 ≤3% on every VQA family); (iii) math peaks on
Fabrication (F5) — **rejected**: math is F4 Fusion (74%), not F5. We report the rejected
hypothesis rather than hide it.

---

## 5. Validity Control: F7 is a Generation-*Interface* Failure

**Objection.** CLIP is handed the label set (closed-set) while generative models answer
open-ended — the gap could be an evaluation artifact.

**Control.** Re-evaluate the generative models with the **candidate labels listed in the
prompt** on the small-label datasets, and judge the residual errors.

**Accuracy (same datasets, labels supplied):**

| Dataset | Gen open-ended | Gen **closed-set** (2B) | Gen closed-set (2.5-3B) | CLIP |
|---|---|---|---|---|
| Food101 | 4.9% | **65.2%** | 69.3% | 96.5% |
| DTD | ~9% | **42.3%** | 39.8% | 82% |
| RESISC45 | 50.4% | **81.6%** | 70.0% | 68.4% |

**Residual failure mode (error even with labels given):**

| Condition | F7 | F6 |
|---|---|---|
| Generative, open-ended | **76–83%** | ~10% |
| Generative, **closed-set** | **22%** | **68%** |
| Contrastive (CLIP) | 22% | 56% |

**Results.** (i) Constraining the output recovers most accuracy (Food101 **4.9→65%**,
RESISC45 **50→82%**), so a large share of the open-ended deficit is *interface*, not
perception. (ii) A **residual deficit persists only on fine-grained tasks** (Food101
closed-set 65% vs CLIP 96%) and vanishes on coarse ones (RESISC45 closed-set 82% ≥ CLIP
68%). (iii) Decisively, **F7 collapses from 76–83% to 22% — exactly the contrastive
baseline — and the residual mode becomes F6, matching CLIP's own 56%.**

**Interpretation.** F7 Overgeneralization is a property of the **free-form generation
interface**, not of the vision backbone or the scoring metric. Unconstrained generation
systematically *discards granularity the model has already perceived* (the contrastive
probe on the identical labels recovers it) — a calibration/informativeness failure that
is invisible to accuracy and directly relevant to whether a VLM's outputs can be trusted.
(Fig. `headline_f7_interface.png`.)

---

## Figures (`results/main2b/figures/`)
`headline_f7_interface.png` · `acc_vs_f7_scatter.png` · `failuremode_by_model.png` ·
`f7_by_architecture.png` · `scale_qwen2vl.png` · `failure_by_family.png` ·
`accuracy_heatmap.png` · `judge_vs_human_kappa.png` · `human_agreement_by_family.png` ·
`human_judge_confusion.png`.
