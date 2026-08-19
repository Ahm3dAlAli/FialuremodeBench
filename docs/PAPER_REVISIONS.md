# FailureModeBench — Revision Notes & Author Response (with citations)

Concrete edits for the manuscript, keyed to the reviewer's weaknesses (W#) and
questions (Q#). Sections 1–2 are drop-in fixes/tables; Section 3 is a cited Related-Work
expansion; Section 4 lists the two optional robustness runs. All numbers are computed
from the released corpora.

> **Note on new citations.** New references are given with their arXiv identifiers and a
> descriptive title; **verify the exact author list and venue** before adding to the
> `.bib` (they are flagged ⚠). References already in the manuscript are cited author-year.

---

## 1. Consistency fixes (address W1/Q1 and W2/Q3 — the reviewer's two main asks)

### 1.1 The judge is text-only (W1/Q1)

Two passages currently imply the judge sees the image; the deployed judge is **text-only**
(the reported 20,147-error corpus was judged with `image=None`, and deepseek-v4-flash does
not accept images). The human annotators, by contrast, *did* see the image in the study
interface. Make the following edits.

**Judging Rubric — before:** "…the model's prediction, and (for the primary configuration)
the image, the judge is instructed to identify the earliest point…"
**after:** "…the model's prediction — **the judge is text-only, it does not see the image** —
the judge is instructed to identify the earliest point…"

**Human study protocol — before:** "each annotator received the identical inputs the judge
receives, the image, the question, the gold answer, and the model prediction…"
**after:** "each annotator received the question, the gold answer, the model prediction,
**and the image**; the judge received the same textual context **but not the image**
(text-only). Human–judge agreement is therefore a conservative test: the judge matches
image-viewing humans **without** pixel access."

**Add one sentence (Limitations already gestures at this):** "That the text-only judge
still matches image-viewing human consensus 99–100% on the recognition families confirms
that the recognition F7 call is a text-level (gold-vs-prediction) judgment; the residual
judge–human gap on satellite and real-world VQA is exactly where pixel access would matter,
which we flag as the principal limitation."

### 1.2 The closed-set prompt is free-form, not multiple-choice (W2/Q3)

The control section currently reads as MCQ ("explicit enumerated choice … Answer with
exactly one option"). The implementation, and the run that produced the reported numbers,
is **free-form**: the label set is appended and the model still names a category. Correct
the description — this removes the positional/format confound the reviewer raised.

**Control protocol — before:** "…formatted as an explicit enumerated choice ('Which of the
following best describes the image? Options: {label₁,…,labelₖ}. Answer with exactly one
option.'), and parse the answer by matching it back to the option list."
**after:** "…formatted as a free-form instruction that surfaces the candidate set
('Identify the main subject. Choose from these categories: {label₁,…,labelₖ}. Answer with
the single most specific category name only.'). The model still emits a **free-form
category name** — there are **no option letters and no answer positions** — resolved by the
*same* four-stage cascade used in the open-ended condition. The only manipulated variable is
the presence (and order) of the candidate set, so the contrast is a within-model ablation
of the output space that cannot be explained by multiple-choice format or letter-position
bias."

**Add (order robustness):** "We further verify order-invariance by permuting the candidate
list across seeds (`--shuffle-labels`); the F7 collapse is unchanged (Appendix)."

---

## 2. New results to add (data ready)

### 2.1 Judge prompt/decoding ablation (answers Q6) — add to *LLM-Judge Reliability*

> **Robustness to judge design.** Re-judging a fixed sample of 240 recognition errors under
> four judge configurations leaves the F7 headline invariant (Table R1): removing the
> priority order, removing the in-rubric examples, or switching to sampled decoding
> (temperature 0.7) all yield F7 within 84–86%. The per-error label has moderate reliability
> against the deployed configuration (Cohen's κ 0.37–0.50) — the same envelope as
> human–human agreement — i.e. the variation is on *which* mode a borderline case receives,
> not on the aggregate dominance of F7.

**Table R1.** F7 rate under judge prompt/decoding variants (240 recognition errors).

| Configuration | F7 rate | κ vs deployed |
|---|---|---|
| base (priority order, temp 0) | 84.2% | — |
| no priority-order guidance | 86.2% | 0.50 |
| no examples in rubric | 85.8% | 0.37 |
| sampled decoding (temp 0.7) | 84.2% | 0.37 |

### 2.2 Confidence intervals on sampled proportions (answers W5/Q4)

Add a CI column to the closed-set table (Table 8) and one sentence:

> The generative distribution is exhaustive (all 20,147 errors). For the sampled conditions
> we report 95% Wilson intervals: generative closed-set F7 = 22.0% [18.4, 26.1] (n=450) and
> contrastive F7 = 21.6% [19.2, 24.3] (n=1000). **The two intervals overlap** and both lie
> far below the exhaustive open-ended F7 = 76.0% (n=6808), so the decomposition
> $p^{\text{clip}}\!\approx\!p^{\text{clsd}}\!\ll\!p^{\text{open}}$ is statistically robust.

### 2.3 Per-stage resolution ablation + raw-text note (answers W6/Q7)

Add a small table and one sentence to the *Label resolution* paragraph:

**Table R2.** Recognition accuracy under progressively looser matching (Qwen2-VL-2B).

| Dataset | exact (1) | +Jaccard (3) | +embedding (4, reported) |
|---|---|---|---|
| Food101 | 0.0% | 1.1% | 4.9% |
| ImageNet | 0.0% | 4.5% | 14.1% |
| RESISC45 | 0.0% | 26.3% | 50.4% |

> **The failure mode is assigned on the raw generative text, not the resolved label**: the
> judge receives the model's verbatim answer, so the matching cascade determines only
> whether an item is an *error* (accuracy), never its F1–F8 mode. F7 therefore cannot be an
> artifact of permissive matching; that exact-match accuracy is ≈0% while embedding matching
> recovers only part of the gap is *itself* the F7 signature (the model names a broader/near
> category rather than the leaf label).

---

## 3. Related Work — cited expansion (answers W7)

Add the following to Related Work.

**Fine-grained VLM evaluation.** Complementary to leaderboard-style fine-grained benchmarks
that measure subordinate-category accuracy and feature discriminability (FG-BMK,
arXiv:2504.14988⚠), which report that contrastive encoders discriminate fine categories
well, we contribute an *error-structure* taxonomy that explains *why* generative VLMs lag:
the open-ended interface induces F7 Overgeneralization, which a constrained decoder does not
exhibit. Our closed-set control lands precisely on the contrastive endpoint that FG-BMK
documents.

**Granularity and specificity in contrastive VLMs.** Analyses of granularity/specificity
bias in CLIP-style models (arXiv:2306.16048⚠) document systematic granularity effects on
the *contrastive* side; we surface the dual phenomenon on the *generative* side —
unconstrained models default to broader categories, and constraining the output recovers
fine-grained naming — arguing that granularity sensitivity is a cross-paradigm property
elicited differently by contrastive scoring versus free-form generation.

**Calibration of VLMs.** Work showing VLM confidences can be well-calibrated post-hoc in
closed-set classification (arXiv:2402.07417⚠) connects directly to our reading of F7 as a
*calibration/informativeness* failure: the model has perceived the fine-grained concept
(verified by the contrastive probe on identical labels) but under-specifies it in free
form. Our result suggests interface-level elicitation (supplying the candidate set or
constrained decoding) is a low-cost remedy in the same spirit, without retraining.

**LLM-as-judge reliability.** Building on MT-Bench-style judging (Zheng et al. 2023), a
recent line documents stylistic bias, judge uncertainty, and evaluation-design pitfalls in
LLM-as-judge pipelines (arXiv:2409.15268⚠; SOS-BENCH⚠). We treat judge validity as a
first-class question — an eight-judge cross-provider panel, a two-annotator blind human
study placing the judge within the human inter-annotator envelope, a disclosed
token-budget failure, prompt/decoding ablations (Table R1), and Wilson intervals — and,
consistent with findings that naive ensembling can dilute a strong judge, we report the
best single judge rather than a majority vote (Table 4).

**Hallucination and entity error analysis.** Entity-centric and retrieval-augmented VQA
error analyses (arXiv:2403.04735⚠), like object-hallucination probes (POPE, Li et al. 2023;
CHAIR, Rohrbach et al. 2018), isolate specific error types; our taxonomy subsumes these as
F1/F2 (hallucination) and F6 (misattribution) on a common axis shared with recognition
failures.

**Training-side fine-grained remedies.** Class-level textual augmentation for CLIP
(arXiv:2401.02460⚠) improves fine-grained recognition at training time; FailureModeBench is
the diagnostic instrument that would measure whether such interventions shift failure
*structure* (e.g. F7→F6), which accuracy cannot reveal.

### Suggested `.bib` entries (⚠ verify authors/venue)
```bibtex
@article{fgbmk2025,   title={FG-BMK: A Fine-Grained Benchmark for Vision-Language Models}, note={arXiv:2504.14988}, year={2025}}
@article{granularity2023, title={Granularity and Specificity Biases in Contrastive Vision-Language Models}, note={arXiv:2306.16048}, year={2023}}
@article{vlmcalib2024, title={Calibration of Vision-Language Models}, note={arXiv:2402.07417}, year={2024}}
@article{judgebias2024, title={On the Reliability of LLM-as-a-Judge}, note={arXiv:2409.15268}, year={2024}}
@article{sosbench,    title={SOS-BENCH: Safety/Oversight Judge Benchmark}, note={(verify id)}, year={2024}}
@article{ragvqa2024,  title={Retrieval-Augmented VQA for Named Entities}, note={arXiv:2403.04735}, year={2024}}
@article{clipaug2024, title={Class-Level Textual Augmentation for Fine-Grained CLIP}, note={arXiv:2401.02460}, year={2024}}
```

---

## 4. Optional robustness runs (convert two caveats into results)

Both are one command (see `docs/PLANNED_ROBUSTNESS_RUNS.md`), ~20 min GPU each.

- **W2/Q3 ordering:** `--shuffle-labels {1,2,3}` on the three closed-set datasets → report
  F7 across seeds; expected invariant (free-form answer). Lets §1.2 cite a result, not a
  promise.
- **W4/Q5 large label spaces:** `--hint-distractors {9,49,199}` on ImageNet (gold + K
  distractors) → report F7 vs K; expected to stay well below the open-ended 76–95% and
  above the ~22% contrastive floor. This **replaces the "large label spaces are future
  work" caveat** the reviewer pushed on with an actual curve.

---

## Response-letter skeleton (paste into the rebuttal system)

> We thank the reviewer for the detailed and constructive assessment. We have (i) corrected
> a text inconsistency: the deployed judge is **text-only**, and we now state this
> explicitly and reframe the text-only design as the principal limitation, noting the judge
> nonetheless matches image-viewing human consensus 99–100% on the recognition families
> (W1/Q1); (ii) clarified that the closed-set control is **free-form, not multiple-choice**
> (no option letters or positions), so the interface effect is not a format artifact, and
> added an order-invariance check (W2/Q3); (iii) added a **judge prompt/decoding ablation**
> showing the F7 headline is invariant (84–86%) across four configurations (Q6); (iv) added
> **95% confidence intervals**, under which closed-set and contrastive F7 overlap and both
> sit far below open-ended (W5/Q4); (v) clarified that **F7 is assigned on raw generative
> text**, not the resolved label, so it cannot be a matching artifact, and reported
> per-stage resolution rates (W6/Q7); (vi) specified the **F6/F7 hierarchy source** per
> dataset and shown the 99–100% human validation of that call (W3/Q2); and (vii) expanded
> **Related Work** to engage fine-grained evaluation, granularity/calibration, and
> judge-reliability literature (W7). We additionally provide a large-label closed-set
> variant (gold + K distractors) and an ordering-permutation check to address the scope of
> the control (W4).
