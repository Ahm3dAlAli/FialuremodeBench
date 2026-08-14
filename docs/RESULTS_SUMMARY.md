# FailureModeBench — Results Summary (paper-ready)

We evaluate **7 models in two paradigms** — five generative VLMs across three
architectures (Qwen2-VL-2B/7B, Qwen2.5-VL-3B, InternVL2.5-8B, LLaVA-1.6-7B) and three
contrastive CLIP-style recognizers (SigLIP2, DFN-CLIP, MetaCLIP2) — over **9 task
families / 16 datasets**, and classify **20,147 errors** with an LLM judge whose validity
we establish against two independent human annotators.

---

## Headline results

**R1 — Aggregate accuracy hides a structured, four-region failure diagnosis.**
Applying one taxonomy across perception and reasoning reveals that each capability fails
in a characteristic way under identical weights (Qwen2-VL-2B, 20,147 errors): recognition
→ **F7 Overgeneralization** (66–99% per family), structured VQA (chart, general) →
**F6 Misattribution** (55–72%), math → **F4 Fusion** (74%), real-world & counterfactual
→ **F1 Visual Hallucination** (40–45%). Accuracy collapses these qualitatively different
behaviors into one number.

**R2 — F7 Overgeneralization is architecture-independent among generative VLMs.**
Every generative model — three architectures, sizes 2B–8B — is F7-dominant in recognition
(66–83%), whereas all three contrastive models are F6-dominant (56%) with F7 at 22%. Scale
lowers F7 (Qwen2-VL 76%→66% from 2B→7B) but never changes the dominant mode.

**R3 — A ~5× accuracy gap between paradigms on the same labels.** Contrastive recognizers
average **80–84%** mean recognition accuracy where generative VLMs average **15–24%**
(20× on fine-grained: 96% vs 4.9%). Sub-billion-parameter CLIP encoders beat 7–8B
generative VLMs — recognition accuracy here is governed by the paradigm, not scale.

**R4 (central) — F7 is a property of the free-form generation *interface*, not of
perception or the metric.** The generative-vs-contrastive gap could be dismissed as
open-set vs closed-set. Our validity control re-runs the generative models **with the
candidate labels in the prompt**: accuracy recovers most of the gap (Food101
4.9%→65.2%, RESISC45 50.4%→81.6%) and **F7 collapses from 76–83% to 22% — exactly the
contrastive baseline** — with the residual mode becoming F6, matching CLIP. Overgeneralization
is a calibration/informativeness failure: the model has perceived the fine-grained concept
(the contrastive probe on identical labels recovers it) but its unconstrained output
discards the granularity.

**R5 — The automated judge is within the human inter-annotator envelope.** Two experts
independently labeled the study blind. Human–human Cohen's κ is **0.471**; the judge's
agreement with each human (**0.428**, **0.552**) brackets it, and adding the judge leaves
Fleiss' κ essentially unchanged (0.462→0.480) — statistically the judge behaves like a
third annotator. On items where both humans agree, the judge matches consensus **85%
overall and 99–100% on the recognition families that carry the finding**; residual
disagreement is localized to satellite land-use (6%) and real-world VQA. We also report an
8-judge cross-provider panel (mean pairwise κ 0.52–0.56, all judges F7-dominant), the most
human-aligned judge (gpt-4o, κ 0.58; the deployed GPU-free deepseek judge 0.47), and a
disclosed harness bug (a reasoning-judge token-budget starvation that had silently left
12.5% of errors unclassified).

---

## Key numbers (drop-in)

| Quantity | Value |
|---|---|
| Errors judged (generative, full) | 20,147 |
| Failure regions | recognition→F7, VQA→F6, math→F4, real/cf→F1 |
| Recognition F7, generative (3 archs) | 66–83% |
| Recognition F7, contrastive | 22% |
| Mean recognition accuracy, generative vs contrastive | ~15–24% vs 80–84% |
| Closed-set control: F7 open→closed | 76–83% → 22% |
| Closed-set control: accuracy (Food101) | 4.9% → 65.2% (CLIP 96.5%) |
| Human–human κ / judge–human κ | 0.471 / 0.428, 0.552 |
| Fleiss κ (2 humans → +judge) | 0.462 → 0.480 |
| Judge vs human consensus | 85% (99–100% on fine-grained/generic/robustness) |
| 8-judge inter-judge κ | 0.52–0.56 |

## One-paragraph abstract-style summary

FailureModeBench diagnoses *how* vision-language models fail, not just *how often*. Across
9 task families and 20,147 errors, a single 8-mode taxonomy exposes four failure regions
under identical weights (recognition→Overgeneralization, VQA→Misattribution, math→Fusion,
real-world→Hallucination). Overgeneralization dominates recognition for every generative
VLM (66–83%) across three architectures but not for contrastive recognizers (22%); a
closed-set control shows this collapses to the contrastive baseline once the output is
constrained, isolating it as a generation-*interface* calibration failure rather than a
perception or scoring artifact. The automated judge underpinning the diagnosis is validated
to lie within the human inter-annotator envelope (judge–human κ 0.43/0.55 vs human–human
0.47; 99–100% agreement with human consensus on the headline families).

---

*Figures: `docs/figures/fig1–8.pdf`. Full setup, per-dataset tables, and the judge panel:
`docs/SETUP_AND_RESULTS.md`. All numbers from the released corpora `results/*/corpus.jsonl`.*
