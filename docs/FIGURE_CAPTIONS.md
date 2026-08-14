# FailureModeBench — Figure captions & explanations

Seven figures in `docs/figures/` (individual PDFs) and `docs/FailureModeBench_figures.pdf`
(combined). Each entry gives a short LaTeX-ready **caption** and a longer **explanation**.

---

## Figure 1 — Recognition accuracy by family (`fig1_accuracy_by_family.pdf`)

**Caption.** Mean recognition accuracy by task family for generative VLMs (n=5) vs
contrastive CLIP-style recognizers (n=3). Contrastive models dominate every family,
most strongly on fine-grained recognition (96% vs 24%).

**Explanation.** Grouped bars compare the two paradigms on the four recognition families.
The gap is largest exactly where the label space is most fine-grained (Fine-grained:
contrastive 96% vs generative 24%; Generic 78 vs 22; Robustness 78 vs 16) and smallest on
coarse land-use classes (Satellite 69 vs 52). This motivates the diagnosis: the deficit is
not uniform, and it is worst where naming the *exact* subclass matters most.

## Figure 2 — Paradigm summary: accuracy and dominant failure mode (`fig2_paradigm_summary.pdf`)

**Caption.** Generative and contrastive recognizers are opposite on both axes: mean
recognition accuracy (28% vs 80%) and the share of errors that are F7 Overgeneralization
(75% vs 22%).

**Explanation.** Two panels summarise the headline contrast. Generative VLMs are far less
accurate *and* fail overwhelmingly by F7 (naming a broader category), whereas contrastive
models are accurate and, when they do err, rarely overgeneralize. Accuracy and failure-mode
structure move together, foreshadowing a common cause (Fig. 4).

## Figure 3 — The two regimes in accuracy–F7 space (`fig3_accuracy_vs_f7.pdf`)

**Caption.** Each model plotted by recognition accuracy vs its F7 rate. Generative VLMs
(red) cluster at low accuracy / high F7; contrastive models (blue) at high accuracy / low
F7. Supplying the candidate labels to a generative model (orange diamond) moves it from the
generative regime to the contrastive one.

**Explanation.** The plane cleanly separates the two paradigms. The orange diamond is the
*closed-set control* (§Validity): the same generative model, given the label set in the
prompt, jumps to ~63% accuracy and 22% F7 — i.e. it crosses into the contrastive regime.
The arrow marks this shift, visually establishing that F7 is tied to the open-ended output
interface rather than the model's perception.

## Figure 4 — F7 is a generation-interface failure (`fig4_interface_collapse.pdf`)

**Caption.** On the same datasets (Food101/DTD/RESISC45): (left) constraining the
generative output to the label set raises accuracy 20%→63%; (right) it collapses F7 from
80% to 22%, the contrastive baseline (−58 points). Overgeneralization is a property of
free-form generation, not perception or the metric.

**Explanation.** The key experiment. When the generative model must choose among the given
labels, most of the accuracy gap closes and the dominant failure mode flips away from F7 to
the contrastive baseline. The residual errors become F6 (sibling misattribution), matching
CLIP. This isolates F7 as an *interface/calibration* failure: the model has perceived the
fine-grained concept but its unconstrained output discards the granularity.

## Figure 5 — Failure modes by model (`fig5_failuremode_by_model.pdf`)

**Caption.** Distribution of recognition errors over F1–F8 for six model groups. Every
generative VLM is F7-dominant (66–83%, labelled); the three contrastive models shift to
F6 Misattribution with F7 at 22%.

**Explanation.** Stacked bars show F7 (red) dominating all five generative models across
three architectures (Qwen2-VL, Qwen2.5-VL, InternVL2.5, LLaVA-1.6) and sizes 2B–8B, while
the CLIP group flips to F6 (blue). Overgeneralization is architecture-independent among
generative VLMs — not a quirk of any one model. Scale lowers F7 (Qwen2-VL 76%→66% from
2B→7B) but does not change the dominant mode.

## Figure 6 — Model size vs recognition accuracy (`fig6_size_vs_accuracy.pdf`)

**Caption.** Mean recognition accuracy vs model size (log scale). Sub-billion-parameter
contrastive encoders (blue) outperform 7–8B generative VLMs (red) by a wide margin.

**Explanation.** Accuracy is plotted against parameter count. The shaded bands highlight
that contrastive models (0.6–0.9B) sit in the high-accuracy region while much larger
generative VLMs (2–8B) sit low — recognition accuracy here is governed by the paradigm, not
scale. Within the generative family, scale helps (Qwen2-VL 2B→7B) but does not close the gap.

## Figure 7 — Failure-mode diagnosis across nine task families (`fig7_family_diagnosis.pdf`)

**Caption.** Per-family distribution of F1–F8 over all 20,147 Qwen2-VL-2B errors. Four
regions emerge under identical weights: recognition → F7, structured VQA → F6, math → F4,
real-world/counterfactual → F1.

**Explanation.** The uniform taxonomy exposes qualitatively different behaviors that
accuracy collapses into one number: the four recognition families are F7-dominated
(Overgeneralization); chart/general VQA are F6-dominated (Misattribution); math is F4
(Fusion — right quantities, wrong combination); real-world and counterfactual are F1
(Visual Hallucination). Each capability fails in a characteristic way.

---

*All numbers computed from the released corpora (`results/*/corpus.jsonl`) and predictions.
Figures are title-less; use these captions in-text.*

## Figure 8 — Judge validity: within the human envelope (`fig8_judge_validity.pdf`)

**Caption.** (Left) Pairwise/Fleiss' κ on the two-annotator study: human–human agreement
(0.47) is bracketed by the judge's agreement with each human (0.43, 0.55), and adding the
judge leaves Fleiss' κ essentially unchanged (0.46→0.48). (Right) On items where both
humans agree, the judge matches human consensus 85% overall and 99–100% on the recognition
families carrying the headline; disagreement localises to satellite and real-world VQA.

**Explanation.** The LLM judge disagrees with a human no more than two humans disagree with
each other — statistically it behaves like a third annotator, validating the automated
taxonomy. Where humans are confident (both agree), the judge is near-perfect on recognition,
and the residual disagreement is a specific, disclosed limitation (satellite land-use).
