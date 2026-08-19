# FailureModeBench — Methodology

## 3.1 Problem formulation and notation

Let a **model** $m$ map an image–query pair $(x, q)$ to an answer $\hat{y} = m(x, q)$,
and let $y^\star$ denote the gold answer. A dataset is a set of examples
$\mathcal{D} = \{(x_i, q_i, y_i^\star)\}_{i=1}^{N}$; each dataset belongs to a **task
family** $\phi \in \Phi$ (e.g. fine-grained recognition, chart VQA, math), and we
group families into two **modalities**: *recognition* (the query is "name the class",
answered against a fixed label set $\mathcal{L}$) and *VQA* (open or multiple-choice
questions). A prediction is **correct** iff $\mathrm{correct}(\hat{y}, y^\star)=1$
under the family's scoring rule (§3.4).

The object of study is not the accuracy $\mathrm{acc}(m,\mathcal{D})=\frac{1}{N}\sum_i
\mathrm{correct}(\hat{y}_i,y_i^\star)$ but the **error set**
$$
\mathcal{E}(m,\mathcal{D}) = \{\, (x_i,q_i,y_i^\star,\hat{y}_i) : \mathrm{correct}(\hat{y}_i,y_i^\star)=0 \,\}.
$$
We introduce a **failure-mode taxonomy** $\mathcal{T}=\{F_1,\dots,F_8\}$ and a
**diagnosis function** $J:\mathcal{E}\to\mathcal{T}$ that assigns each error exactly
one mode. FailureModeBench reports, per family, the induced distribution
$$
p_\phi(F) \;=\; \frac{1}{|\mathcal{E}_\phi|}\sum_{e\in\mathcal{E}_\phi}\mathbb{1}\!\left[J(e)=F\right],
\qquad F\in\mathcal{T},
$$
i.e. *how* a capability fails, conditional on it having failed — a quantity that
$\mathrm{acc}$ marginalises away.

## 3.2 The failure-mode taxonomy $\mathcal{T}$

The eight modes are defined to be **mutually exclusive** (one label per error) and
**modality-agnostic** (the same $\mathcal{T}$ applies to perception and reasoning),
which is what lets us compare *where* different capabilities fail on a common axis:

| | Mode | Formal criterion (what distinguishes it) |
|---|---|---|
| $F_1$ | Visual Hallucination | $\hat{y}$ asserts a visual entity $\notin x$ |
| $F_2$ | Linguistic Hallucination | $\hat{y}$ asserts a claim unsupported by $(x,q)$, not about a missing object |
| $F_3$ | Cross-Modal Misalignment | $\hat{y}$ is independent of $x$ (ignores/mis-binds the image) |
| $F_4$ | Fusion Error | components of $\hat{y}$ are individually grounded but combined wrongly |
| $F_5$ | Fabrication | $\hat{y}$ introduces a step/relation with no support in $(x,q)$ |
| $F_6$ | Misattribution | correct property assigned to the **wrong** entity/class ($\hat{y}$ a sibling of $y^\star$) |
| $F_7$ | Overgeneralization | $\hat{y}$ is a **hypernym** of $y^\star$: $\hat{y}\succ y^\star$ in the label hierarchy |
| $F_8$ | Confabulation | $\hat{y}$ is fluent and internally coherent but its conclusion is false |

The $F_6/F_7$ distinction is central and formalised via the label hierarchy $\succ$:
$F_7$ requires $\hat{y}\succ y^\star$ (a *broader, correct* category — "fish" for
"tench"), whereas $F_6$ is a lateral confusion ($\hat{y}$ and $y^\star$ share a
parent — "donut" for "beignet"). The assignment applies a fixed **priority order**
(check $F_7$; then $F_1/F_2$; then $F_4$; then $F_5$; then $F_6$; then $F_8$; else
$F_3$) so that $J$ is a total function.

## 3.3 Two evaluation paradigms

We evaluate two model classes that differ precisely in their **output space**, which
is the independent variable of our central experiment (§3.6).

**Generative VLMs.** $m_{\text{gen}}(x,q)$ emits free text over an unbounded
vocabulary $\mathcal{V}^\ast$; for recognition the query is open-ended ("name the most
specific category"), so $\hat{y}\in\mathcal{V}^\ast$ and must be *resolved* to the
label set post hoc (§3.4).

**Contrastive (CLIP-style) recognizers.** $m_{\text{clip}}$ is a dual encoder
$(f_{\text{img}}, f_{\text{txt}})$; zero-shot classification is the constrained
$\arg\max$ over the label set,
$$
\hat{y} \;=\; \arg\max_{\ell\in\mathcal{L}} \; \cos\!\big(f_{\text{img}}(x),\; \bar t_\ell\big),
\qquad
\bar t_\ell = \frac{1}{|P|}\sum_{\pi\in P} \frac{f_{\text{txt}}(\pi(\ell))}{\lVert\cdot\rVert},
$$
with a small prompt-template ensemble $P$. Crucially $\hat{y}\in\mathcal{L}$ **by
construction** — a contrastive model *cannot emit an out-of-set answer*, and in
particular cannot produce a hypernym $\hat{y}\succ y^\star$ unless that hypernym is
itself a label in $\mathcal{L}$. This structural asymmetry in the output space is
what we exploit to localise the mechanism of $F_7$.

## 3.4 Answer resolution and scoring (recognition)

Generative free-text answers are mapped to $\mathcal{L}$ by a deterministic,
order-of-precedence **matching cascade** $R:\mathcal{V}^\ast\to\mathcal{L}\cup\{\bot\}$:
(i) exact substring / whole-phrase match; (ii) a curated synonym table; (iii) Jaccard
token-overlap above a threshold; (iv) nearest label by sentence-embedding cosine
(the CLIP-benchmark standard). $\mathrm{correct}(\hat{y},y^\star)=\mathbb{1}[R(\hat{y})=y^\star]$.
Because $R$ becomes progressively more permissive, the **sensitivity of accuracy to
the resolution stage is itself diagnostic**: under exact matching (stage i)
generative recognition accuracy is $\approx 0$ — the model rarely emits the clean
label token — and rises only once semantic matching credits near-misses. This gap is
the quantitative signature of $F_7$ and motivates reporting the matcher explicitly.
For VQA we defer to VLMEvalKit's per-dataset extraction/scoring.

## 3.5 Automated failure judging

The diagnosis function $J$ is realised by an LLM **judge** $g_\theta$ prompted with a
fixed rubric $\rho$ (the definitions of §3.2 and the priority order) and the tuple
$(q, y^\star, \hat{y}, \phi)$; it returns a structured verdict
$g_\theta(\rho, q, y^\star, \hat{y}) = (F, c, r)$ with mode $F\in\mathcal{T}$,
confidence $c\in[0,1]$, and rationale $r$. The judge is **text-only** by design
(GPU-free deployment): it reasons over the gold/prediction pair, not the pixels — a
deliberate scope choice we validate in §3.7 and revisit in Limitations. Two
methodological points:

- **Determinism / parsing.** We decode at temperature $0$ and parse strict JSON;
  a verdict is retried up to $k$ times if unpardeable. A practical caveat we report:
  a *reasoning* judge spends output tokens on a hidden reasoning pass, so an
  insufficient token budget truncates before the JSON and silently yields "NA";
  the budget must be set accordingly (we use 2048).
- **Enrichment.** For multiple-choice items the bare gold letter is uninformative;
  we substitute the resolved option text so $g_\theta$ sees the semantic answer.

## 3.6 The closed-set validity control

The naive comparison "generative $\mathrm{acc}\approx$ 15% vs contrastive
$\approx$ 80%, and $p^{\text{gen}}(F_7)\gg p^{\text{clip}}(F_7)$" confounds two
factors: the **model** and the **output space** (open-ended vs closed-set). To
isolate them we hold the model fixed and intervene on the output space only.

For each small-label dataset we run the *same* generative model in a **closed-set**
condition: the prompt is augmented with the candidate labels
$q' = q \oplus \mathcal{L}$, restricting the intended output to $\mathcal{L}$ while the
weights, image, and everything else are unchanged. Crucially the **answer modality is
held fixed**: the model still produces a *free-form* category name (resolved by the
same cascade $R$), not a multiple-choice letter — there are no answer positions or
option identifiers, so letter-position bias cannot arise. The only manipulated variable
is the presence (and order) of the candidate set; we test order-invariance by permuting
$\mathcal{L}$ across seeds. Writing
$\mathrm{acc}^{\text{open}}, \mathrm{acc}^{\text{closed}}$ and $p^{\text{open}}(F_7),
p^{\text{closed}}(F_7)$ for the two conditions, the control estimates the
**interface-attributable** component of the effect:
$$
\Delta_{\text{acc}} = \mathrm{acc}^{\text{closed}} - \mathrm{acc}^{\text{open}},
\qquad
\Delta_{F_7} = p^{\text{open}}(F_7) - p^{\text{closed}}(F_7).
$$
If $F_7$ were a *perception* failure it would be invariant to the output constraint
($\Delta_{F_7}\approx 0$); if it is an *interface* failure, constraining the output
should collapse it toward the contrastive baseline
($p^{\text{closed}}(F_7)\approx p^{\text{clip}}(F_7)$). The residual mode under
closed-set decoding then identifies what remains after the interface effect is
removed. This is the paper's central identification strategy.

## 3.7 Judge validity: agreement notions

Because every result rests on $J\!=\!g_\theta$, we treat the judge as a measurement
instrument and quantify its validity against human annotators using standard
chance-corrected agreement.

**Cohen's $\kappa$** for two raters $a,b$ over items with observed agreement $p_o$ and
chance agreement $p_e=\sum_{F}\hat p_a(F)\hat p_b(F)$:
$$
\kappa \;=\; \frac{p_o - p_e}{1 - p_e}.
$$
We report $\kappa$ for every pair in $\{\text{human}_1,\text{human}_2,\text{judge}\}$.

**Fleiss' $\kappa$** for $n$ raters over $N$ items (agreement beyond chance across the
panel), computed on the subset labeled by all raters. The key **validity notion** is
whether the judge lies **within the human inter-annotator envelope**: we compare
$\kappa(\text{human}_1,\text{human}_2)$ against $\kappa(\text{human}_i,\text{judge})$,
and check that adding the judge to the human panel leaves Fleiss' $\kappa$ essentially
unchanged — i.e. the judge is statistically exchangeable with a human annotator.

**Consensus agreement.** On the high-confidence subset where the humans agree
($\text{human}_1=\text{human}_2$), we report the judge's match rate
$\Pr[g_\theta=\text{human} \mid \text{humans agree}]$, per family — a stricter test
that isolates whether the judge is correct where the ground truth is least ambiguous.

**Inter-judge robustness.** Independently, an 8-judge cross-provider panel bounds the
sensitivity of $p_\phi(F)$ to the choice of $g_\theta$ (mean pairwise $\kappa$ and the
per-panel dominant-mode agreement).

**Ground-truth noise.** Annotators may flag an error as *gold-wrong* (the dataset
label is incorrect and the model was right); these are excluded from $\kappa$ and
reported separately as a label-noise rate, since they are not model failures.

---

*Symbols: $x$ image, $q$ query, $y^\star$ gold, $\hat y$ prediction, $\mathcal{L}$
label set, $\mathcal{T}=\{F_1..F_8\}$ taxonomy, $J$ diagnosis, $g_\theta$ LLM judge,
$R$ matching cascade, $\succ$ label-hierarchy hypernymy, $p_\phi(F)$ per-family
failure distribution.*
