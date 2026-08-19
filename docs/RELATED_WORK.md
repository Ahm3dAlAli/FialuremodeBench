# Related Work

**Fine-grained VLM evaluation.** Recent fine-grained benchmarks (e.g. FG-BMK,
arXiv:2504.14988) measure subordinate-category performance and feature discriminability,
and report that *contrastive* encoders excel at fine-grained discrimination. Our work is
complementary: rather than another accuracy leaderboard, we provide an **error-structure
taxonomy** that diagnoses *how* failures occur, and we explain *why* generative VLMs lag
on fine-grained recognition — the open-ended interface induces F7 Overgeneralization, which
a constrained (contrastive or closed-set) decoder does not exhibit. FG-BMK's finding that
contrastive models discriminate well is exactly the endpoint our closed-set control lands
on (F7 → the contrastive baseline).

**Granularity and specificity biases in contrastive VLMs.** Prior analyses of granularity
and specificity (e.g. arXiv:2306.16048) document systematic granularity effects in
contrastive models. We surface a distinct but related phenomenon on the *generative* side:
when unconstrained, generative VLMs default to broader categories, and constraining the
output space recovers fine-grained naming. Together these results argue that granularity
sensitivity is a cross-paradigm property elicited differently by contrastive scoring vs
free-form generation.

**Calibration of VLMs.** Work on VLM calibration (e.g. arXiv:2402.07417) shows that
closed-set classification can be well-calibrated with simple post-hoc scaling. This
connects directly to our interpretation of F7 as a *calibration/informativeness* failure:
the model perceives the fine-grained concept (verified by the contrastive probe on the same
labels) but its unconstrained output *under-specifies* it. Our closed-set result suggests
that interface-level elicitation — supplying the candidate set or constrained decoding — is
a low-cost remedy in the same spirit as post-hoc calibration, without retraining.

**LLM-as-judge reliability.** A growing literature raises concerns about stylistic biases,
uncertainty, and evaluation design in LLM-as-judge pipelines (e.g. arXiv:2409.15268;
SOS-BENCH and related judge-uncertainty/design studies). We treat judge validity as a
first-class question: an 8-judge cross-provider panel bounds sensitivity to judge identity,
a two-annotator blind human study places the judge *within the human inter-annotator
envelope*, and we disclose an evaluation pitfall (reasoning-judge token-budget starvation).
Motivated by this literature we additionally add prompt/decoding ablations and report
confidence intervals; consistent with findings that ensembling can dilute a strong judge,
we report the single best judge rather than a naive majority vote.

**Hallucination and entity error analysis.** Entity-centric and retrieval-augmented VQA
work (e.g. arXiv:2403.04735) analyses hallucination and tail-entity errors. Though
orthogonal in method, its move from aggregate scores toward error-aware assessment shares
our motivation; our taxonomy provides a unified account in which such errors map to F1/F2
(hallucination) and F6 (misattribution), placed on the same axis as recognition failures.

**Training-side fine-grained remedies.** Class-level textual augmentation for CLIP (e.g.
arXiv:2401.02460) improves fine-grained recognition on the training side. FailureModeBench
is the natural **diagnostic instrument** to measure how such interventions reshape failure
*structure* — e.g. whether they shift errors from F7 (overgeneralization) toward F6
(sibling confusion), which accuracy alone cannot reveal.

**Positioning.** Existing suites either (i) report aggregate accuracy, (ii) target a single
failure type (hallucination), or (iii) study one paradigm. FailureModeBench contributes a
*unified, modality-agnostic* failure taxonomy applied across perception and reasoning, an
LLM judge validated to human-annotator agreement, and — via the closed-set control — a
*mechanistic* separation of interface-induced from perceptual failure. This makes it an
alignment-oriented auditing tool (behavioral failure structure, judge validity as a
measurement question) rather than a capability leaderboard.
