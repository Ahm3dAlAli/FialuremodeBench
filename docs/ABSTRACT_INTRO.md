# FailureModeBench — Abstract & Introduction (draft)

*Target: AAAI — AIA: Evaluation Validity (secondary: Evaluation, Auditing & Assurance).*

---

## Abstract

Accuracy tells us *that* a vision-language model (VLM) is wrong; deploying VLMs
safely requires knowing *how* and *why*. We present **FailureModeBench**, a diagnosis
framework that routes every incorrect VLM prediction to one of eight interpretable
failure modes (F1–F8) — spanning hallucination, misattribution, overgeneralization,
fusion, and fabrication — and applies this single taxonomy uniformly across nine task
families (fine-grained and generic recognition, robustness, satellite, chart/diagram,
general and real-world VQA, math, and counterfactual reasoning). Classifying
**20,000+ errors from seven models** with an LLM judge whose validity we scrutinize
(an eight-judge cross-provider panel, a blind human study, and a disclosed harness
bug that had silently left 12.5% of errors unclassified), we find that a two-number
accuracy story hides a structured, family-dependent diagnosis: recognition errors are
dominated by **Overgeneralization (F7)**, structured VQA by **Misattribution (F6)**,
math by **Fusion (F4)**, and counterfactual/real-world by **Hallucination (F1)**.
Our central result is a validity control that isolates the mechanism of the
recognition failure. Naively, generative VLMs score ~15% where contrastive
(CLIP-style) models score ~80% on the same labels — but this compares open-ended
generation against closed-set classification. When we re-run the generative models
**with the candidate labels in the prompt**, F7 collapses from **76–83% to 22%** —
exactly the contrastive baseline — and the residual error mode becomes **F6, matching
CLIP's own residual**. Overgeneralization is therefore a property of the *free-form
generation interface*, not of perception or the scoring metric: unconstrained
generation systematically discards granularity the model has already perceived, a
calibration/informativeness failure invisible to accuracy. We release the taxonomy,
the GPU-free automated judge, and the full per-error labels, turning opaque accuracy
gaps into an actionable behavioral audit for VLM alignment.

---

## 1. Introduction

**Accuracy is the wrong unit for alignment.** Two vision-language models with the
same accuracy can be entirely different objects from a safety standpoint: one that
errs by *hallucinating* content that is not present is a different deployment risk
than one that errs by naming a *broader but correct* category. Yet the field
evaluates VLMs almost entirely by aggregate accuracy, which cannot distinguish these
behaviors. For alignment — deciding when to trust a model's output, where to place
guardrails, how to calibrate downstream use — what matters is the *structure* of the
errors, not their rate.

**We diagnose, not just score.** FailureModeBench assigns every wrong prediction to
one of eight interpretable failure modes (F1–F8), and — unusually — applies the *same*
taxonomy across both perception (recognition) and reasoning (VQA, math,
counterfactual) tasks, over nine families. This uniformity lets us ask where in the
taxonomy each capability fails. It reveals four distinct regions under identical model
weights: recognition fails by **F7 Overgeneralization**, structured VQA by **F6
Misattribution**, math by **F4 Fusion**, and counterfactual/real-world reasoning by
**F1 Visual Hallucination**. The single "accuracy" number hides both this structure
and its near-uniform *cause* within each region.

**We take the evaluator itself as an object of scrutiny.** Because the diagnosis rests
on an LLM judge, its validity is the paper's crux, not a footnote. We report an
eight-judge cross-provider panel (mean pairwise Cohen's κ 0.55, unanimous on the
dominant recognition mode), a blind human study, an explicit dataset-mislabel rate,
and a harness bug we found and fixed — a reasoning-judge token-budget starvation that
had silently left 12.5% of errors unclassified. Trustworthy automated evaluation *of*
failure behavior is itself an alignment contribution.

**Our central finding isolates a mechanism, not a correlation.** The most striking
raw number is that generative VLMs score ~15% on recognition where contrastive
(CLIP-style) recognizers score ~80% on the *same* label sets, and the generative
errors are 76–99% F7 Overgeneralization while contrastive errors are 22%. A skeptic
rightly objects that this compares open-ended generation to closed-set classification.
We run the control: re-evaluating the generative models **with the candidate labels
supplied in the prompt**, F7 collapses from 76–83% to **22% — precisely the
contrastive baseline** — while accuracy recovers most of the gap (e.g. Food101
4.9%→65.2%) and the residual error mode becomes **F6 sibling-misattribution, matching
what CLIP does**. This localizes overgeneralization exactly: it is a property of the
*free-form generation interface*, not of the vision backbone or the metric. Framed for
alignment, F7 is a **calibration/informativeness failure** — the model perceives the
fine-grained concept (as the contrastive probe on identical labels confirms) but its
unconstrained output systematically *under-reports* it.

**Contributions.**
1. A unified eight-mode failure taxonomy applied across perception *and* reasoning,
   over nine task families and 20,000+ judged errors from seven models spanning three
   generative architectures (Qwen2-VL, InternVL2.5, LLaVA-1.6) and three contrastive
   encoders (SigLIP2, DFN-CLIP, MetaCLIP2).
2. The finding — with a closed-set validity control — that F7 Overgeneralization is a
   generation-*interface* failure that collapses to the contrastive baseline once the
   output is constrained, isolating it from perception and the metric.
3. A validated, GPU-free automated failure judge and the complete labeled error corpus,
   released for reuse as a behavioral audit for VLM alignment.

---

*Honest gaps to close before submission (tracked): the human study is currently
single-annotator (κ 0.44); we are collecting ≥2 additional annotators for Fleiss' κ.*
