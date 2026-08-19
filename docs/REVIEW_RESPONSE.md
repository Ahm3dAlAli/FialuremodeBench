# Response to Reviewer Feedback

We address each concern below. **[Fixed]** = clarified/corrected in the text or code;
**[New analysis]** = additional result computed from the released corpora; **[Planned
run]** = a robustness experiment now wired into the harness (one command) to be added.

---

## W1 / Q1 — Did the judge see images? (text-only vs image-conditioned) **[Fixed]**

The deployed judge is **text-only**, and all reported corpora were produced text-only.
The confusion came from a legacy default in `judge.py`; we have set
`attach_images=False` as the default and documented it. Concretely, the full 20,147-error
judging (`expand_corpus.py`) calls the provider with `image=None`, and the deployed
model (deepseek-v4-flash) does not accept image input. The judge reasons over
$(q, y^\star, \hat y, \varphi)$ — the question, gold, prediction, and family.

**Implication we now state explicitly.** Because the judge does not see pixels, F1
(Visual Hallucination) and F6 (Misattribution) on VQA are inferred from the
gold/prediction mismatch and the question, not from the image. This is exactly why the
judge–human gap is largest on the image-heavy families (satellite, real-world VQA;
consensus accuracy 6% / 25%), which we now foreground as the principal limitation. The
recognition headline is unaffected: it does not require pixel access (gold vs a
free-text class name), and there the judge matches human consensus 99–100%.

## W2 / Q3 — Is the "interface" claim confounded by MCQ format / positional bias? **[Fixed + Planned run]**

**Clarification that removes most of the confound:** the closed-set condition is **not**
multiple-choice. The prompt appends *"Choose from these categories: {list}"* and the
model still answers **free-form** (a category name), resolved by the *same* cascade used
in the open condition. There are **no A/B/C/D letters and no answer positions**, so
letter-position bias cannot arise. The only remaining format change is the presence and
*ordering* of the candidate list — which is precisely the intervention
$\mathrm{do}(\mathcal{O}=\mathcal{L})$ we intend (surface the output set), holding weights,
image, family, and answer modality (free text) fixed.

**[Planned run] Ordering robustness.** To rule out list-ordering effects we added
`--shuffle-labels SEED`; we will report F7 across ≥3 random label orderings and show the
collapse is invariant to order. (We expect invariance since the answer is free-form, not
positional.)

## W3 / Q2 — How are F6/F7 hypernym/sibling relations instantiated (DTD, RESISC45)? **[Fixed]**

We do not supply an external ontology; the **judge infers hypernymy from label
semantics** given the rubric's operational test ("is $\hat y$ a strictly broader but
correct category than $y^\star$?"). For ImageNet the labels are WordNet synsets, so this
coincides with WordNet hypernymy; for DTD (textures) and RESISC45 (land-use) no standard
hierarchy exists and the judge reasons over the label strings.

**Why this is not a circularity that invents F7:** (i) the human study directly tests the
F6/F7 decision — on the recognition families where humans agree, the judge matches **99–
100%**, so the hypernym/sibling call is human-validated, not a private LLM artifact; (ii)
F7 is corroborated by an independent, ontology-free signal — the matching-cascade gap
(§W6): the model's raw answers are semantically near but lexically broader than the gold.
We now state the hierarchy source per dataset and the validation explicitly.

## W4 / Q5 — Closed-set only on small-label sets; does F7 collapse at large label spaces? **[Planned run]**

Correct that the control used Food101/DTD/RESISC45 (label lists that fit a prompt). We
added `--hint-distractors K`: for large sets (ImageNet-1k) each item is prompted with the
**gold + K random distractors** (shuffled), a closed-set condition at controlled
difficulty $K\in\{9,49,199\}$. This tests the reviewer's exact question — whether the F7
collapse persists as the candidate set grows — without listing 1000 labels. We will
report F7 vs $K$; the hypothesis is a graceful degradation of the collapse with $K$,
bounded below by the contrastive baseline.

## W5 / Q4 — Sampling, CIs, stratification for contrastive & closed-set judging **[New analysis]**

The **generative** distribution is exhaustive (all 20,147 errors judged — no sampling).
The **contrastive** (n=1000) and **closed-set** (n=450) sets are stratified per family;
we now report 95% Wilson intervals:

| Condition | F7 | 95% CI |
|---|---|---|
| Generative, open-ended (exhaustive, n=6808) | 76.0% | — |
| Generative, closed-set (n=450) | 22.0% | [18.4, 26.1] |
| Contrastive / CLIP (n=1000) | 21.6% | [19.2, 24.3] |

The closed-set and contrastive F7 intervals **overlap** (statistically indistinguishable),
while the open-set value lies far outside both — the decomposition
$p^{\text{clip}}\!\approx\!p^{\text{clsd}}\!\ll\!p^{\text{open}}$ is significant, not a
sampling artifact.

## W6 / Q7 — Could the resolution cascade make F7 a matcher artifact? **[New analysis]**

Two facts rule this out. **(1) F7 is judged on the raw generative text**, not the resolved
label: the judge receives `prediction` (the model's verbatim answer), so the matching
stage cannot influence the F7 assignment. The cascade only affects whether an item is an
*error* (accuracy), not its failure mode. **(2)** The per-stage ablation shows the errors
are genuine, not resolution failures — accuracy stays low even under the loosest matcher,
and exact-match ≈0% is itself the F7 signature:

| Dataset | stored (embed) | +Jaccard | exact |
|---|---|---|---|
| Food101 | 4.9% | 1.1% | 0.0% |
| ImageNet | 14.1% | 4.5% | 0.0% |
| RESISC45 | 50.4% | 26.3% | 0.0% |

The model almost never emits the clean label token (exact ≈0), consistent with naming a
broader/near category; embeddings recover only part of it. We now report per-stage rates
and note that F7 uses raw text.

## Q6 — Judge prompt/decoding ablations (secondary mode, CoT, sampling) **[New analysis]**

We re-judged a fixed sample of 240 recognition errors under four judge configurations
and measured the F7 rate and agreement with the deployed config:

| Judge configuration | F7 rate |
|---|---|
| base (deployed: priority order, temp 0) | 84.2% |
| no priority-order guidance | 86.2% |
| no examples in the rubric | 85.8% |
| sampled decoding (temp 0.7) | 84.2% |

**The F7 headline is invariant** (84–86%, a ~2-point range) to removing the priority
order, removing the in-rubric examples, and switching from deterministic to sampled
decoding. The per-error label has only moderate reliability against the base
configuration (Cohen's κ 0.37–0.50, raw 82–87%) — the same envelope as human–human and
human–judge agreement — but this variation is on *which* mode a borderline case gets, not
on the aggregate dominance of F7. Robustness to the *choice of judge model* is
independently quantified by the 8-judge cross-provider panel (mean pairwise κ 0.52–0.56;
all eight rank F7 dominant). (Script: `scripts/judge_ablation.py`.)

## W7 — Related work / positioning **[Fixed]**

Added a Related Work section (`docs/RELATED_WORK.md`) engaging: fine-grained VLM
evaluation (FG-BMK), granularity/specificity biases in contrastive VLMs, VLM calibration
(linking F7 to under-specification/miscalibration), LLM-as-judge reliability (stylistic
bias, uncertainty, design), entity-hallucination/RAG-VQA error analysis, and class-level
textual augmentation for CLIP (positioning FailureModeBench as the diagnostic suite that
would measure whether such training shifts F7→F6).

---

## Summary of changes
- **Text-only judging** made the code default + stated as the principal limitation (W1/Q1).
- **Closed-set clarified as free-form** (not MCQ), removing the positional-bias confound;
  `--shuffle-labels` added for the ordering check (W2/Q3).
- **Hierarchy source + human validation** of F6/F7 stated per dataset (W3/Q2).
- **`--hint-distractors`** added for the large-label closed-set variant (W4/Q5).
- **Wilson CIs** for all sampled proportions; overlap of closed-set vs contrastive F7 (W5/Q4).
- **F7 shown independent of the matcher** (judged on raw text) + per-stage ablation (W6/Q7).
- **Related Work** added (W7); judge-choice robustness bounded by the 8-judge panel; **judge prompt/decoding ablation done** (F7 84-86% invariant across 4 configs) (Q6).
