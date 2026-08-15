# FailureModeBench — Mathematical Formulation

A compact, self-contained formalisation of the whole pipeline: from raw predictions
to the per-family failure distribution, the interface-identification result, and the
validity estimators.

---

## 1. Spaces and objects

- Images $x\in\mathcal{X}$, queries $q\in\mathcal{Q}$, answers in a token space
  $\mathcal{V}^\ast$. A gold answer is $y^\star\in\mathcal{Y}$.
- A dataset $\mathcal{D}=\{(x_i,q_i,y_i^\star)\}_{i=1}^{N}\sim P_\mathcal{D}$, with a
  family map $\varphi:\mathcal{D}\to\Phi$ (9 families) and a modality
  $\mu(\varphi)\in\{\text{rec},\text{vqa}\}$. Recognition families carry a finite
  label set $\mathcal{L}_\mathcal{D}\subset\mathcal{Y}$.
- A model is a map $m:\mathcal{X}\times\mathcal{Q}\to\mathcal{V}^\ast$, $\hat y=m(x,q)$.
- Taxonomy $\mathcal{T}=\{F_1,\dots,F_8\}$; label hierarchy $(\mathcal{Y},\succ)$ with
  $u\succ v$ meaning "$u$ is a (strict) hypernym of $v$".

## 2. Correctness and the error distribution

Define the family scoring rule $\mathrm{corr}:\mathcal{V}^\ast\times\mathcal{Y}\to\{0,1\}$
(for recognition via the resolver $R$, Eq. 6). The **error indicator** and the
**error (sub)distribution** are

$$
\varepsilon(x,q,y^\star;m)=1-\mathrm{corr}\!\big(m(x,q),y^\star\big),
\qquad
P^{\text{err}}_{m,\varphi}(\cdot)=P_\mathcal{D}\!\big(\cdot \mid \varphi,\ \varepsilon=1\big).
\tag{1}
$$

The benchmark studies $P^{\text{err}}$, **not** the scalar
$\mathrm{acc}(m,\mathcal{D})=\mathbb{E}_{P_\mathcal{D}}[\,1-\varepsilon\,]$.

## 3. The diagnosis operator and the estimand

A **diagnosis operator** assigns one mode to each error,

$$
J:\ \underbrace{\mathcal{X}\times\mathcal{Q}\times\mathcal{Y}\times\mathcal{V}^\ast}_{\text{error tuple }e}\ \longrightarrow\ \mathcal{T},
\tag{2}
$$

realised as a priority-ordered decision list over predicates $\pi_F(e)$ (§3.2), so
$J$ is total and single-valued:
$J(e)=F_{k^\ast}$, $k^\ast=\min\{k:\pi_{F_{(k)}}(e)=1\}$. The **estimand** is the
per-family failure distribution (a point on the simplex $\Delta^{7}$)

$$
p_\varphi(F)\;=\;\Pr\nolimits_{e\sim P^{\text{err}}_{m,\varphi}}\!\big[J(e)=F\big],
\qquad F\in\mathcal{T}.
\tag{3}
$$

Its plug-in estimator over the observed error set $\mathcal{E}_\varphi$ is
$\hat p_\varphi(F)=|\mathcal{E}_\varphi|^{-1}\sum_{e\in\mathcal{E}_\varphi}\mathbb{1}[J(e)=F]$.

## 4. Paradigms as an output-space constraint

The two model classes are identical except for the **admissible output set**
$\mathcal{O}(x,q)\subseteq\mathcal{V}^\ast$ from which $\hat y$ is drawn:

$$
\textbf{generative:}\ \ \mathcal{O}_{\text{gen}}=\mathcal{V}^\ast,
\qquad\qquad
\textbf{contrastive:}\ \ \hat y=\arg\max_{\ell\in\mathcal{L}_\mathcal{D}}\ \big\langle f_{\text{img}}(x),\,\bar t_\ell\big\rangle,\ \ \mathcal{O}_{\text{clip}}=\mathcal{L}_\mathcal{D},
\tag{4}
$$

with $\bar t_\ell=\mathrm{norm}\big(|P|^{-1}\sum_{\pi\in P} f_{\text{txt}}(\pi(\ell))\big)$.
**Structural fact (no-hypernym lemma).** If $\hat y\in\mathcal{O}=\mathcal{L}_\mathcal{D}$
then $\Pr[\,\hat y\succ y^\star\,\wedge\,\hat y\notin\mathcal{L}_\mathcal{D}\,]=0$; i.e.
a constrained model can only overgeneralise *into* the label set. This is the lever
of §7.

## 5. Recognition scoring: the matching cascade

Free text is resolved to a label by a composition of increasingly permissive
partial maps $R=R_4\circ R_3\circ R_2\circ R_1$ (substring $\to$ synonym $\to$ Jaccard
$\to$ embedding-NN), $R:\mathcal{V}^\ast\to\mathcal{L}_\mathcal{D}\cup\{\bot\}$:

$$
\mathrm{corr}(\hat y,y^\star)=\mathbb{1}\!\big[R(\hat y)=y^\star\big],
\qquad
\mathrm{acc}^{(j)}=\mathbb{E}\big[\mathbb{1}[(R_j\circ\cdots\circ R_1)(\hat y)=y^\star]\big].
\tag{6}
$$

The monotone gap $\mathrm{acc}^{(4)}-\mathrm{acc}^{(1)}$ (embedding minus exact) is
itself a scalar signature of $F_7$: it is large precisely when the model names
semantically-near but lexically-distinct (typically broader) labels.

## 6. The judge as a plug-in estimator of $J$

$J$ is unobserved; we estimate it with an LLM judge
$g_\theta(\rho,e')=(\hat F,c,r)$, $\hat F\in\mathcal{T}$, on a reduced tuple
$e'=(q,y^\star,\hat y,\varphi)$ (text-only; pixels excluded by design). Decoding is
deterministic (temp $0$), output parsed as strict JSON, retried $\le k$ times. The
plug-in estimate of Eq. (3) is

$$
\hat p^{\,g}_\varphi(F)=\frac{1}{|\mathcal{E}_\varphi|}\sum_{e\in\mathcal{E}_\varphi}\mathbb{1}\big[g_\theta(\rho,e')=F\big].
\tag{7}
$$

**Consistency assumption (A1).** $g_\theta$ is an unbiased-enough surrogate:
$\mathbb{E}\,\mathbb{1}[g_\theta=F]\approx p_\varphi(F)$, an assumption we do not take
for granted but *test* in §8 (validity) and §9 (inter-judge robustness).

## 7. Identification: isolating the interface effect

The raw contrast confounds **model** and **output space**. Fix the generative model
$m_{\text{gen}}$ and apply an intervention $\mathrm{do}(\mathcal{O}=\mathcal{L})$ that
constrains *only* the output space — operationally, augment the prompt with the label
set, $q'=q\oplus\mathcal{L}_\mathcal{D}$, holding weights, image, and family fixed.
Write the two potential outcomes

$$
p^{\text{open}}_\varphi(F)=p_\varphi\!\big(F\mid \mathrm{do}(\mathcal{O}=\mathcal{V}^\ast)\big),
\qquad
p^{\text{clsd}}_\varphi(F)=p_\varphi\!\big(F\mid \mathrm{do}(\mathcal{O}=\mathcal{L})\big).
\tag{8}
$$

Define the **interface effect** on overgeneralisation and accuracy

$$
\Delta_{F_7}=p^{\text{open}}_\varphi(F_7)-p^{\text{clsd}}_\varphi(F_7),
\qquad
\Delta_{\text{acc}}=\mathrm{acc}^{\text{clsd}}-\mathrm{acc}^{\text{open}}.
\tag{9}
$$

**Hypotheses.** (H$_0$, perception) $F_7$ is invariant to $\mathcal{O}$:
$\Delta_{F_7}\approx0$. (H$_1$, interface) constraining $\mathcal{O}$ collapses $F_7$
to the contrastive baseline: $p^{\text{clsd}}_\varphi(F_7)\approx p^{\text{clip}}_\varphi(F_7)$.
The estimand of interest is the **decomposition**

$$
\underbrace{p^{\text{clip}}_\varphi(F_7)}_{\text{residual (in-set) overgen.}}
\ \le\
\underbrace{p^{\text{clsd}}_\varphi(F_7)}_{\text{constrained generative}}
\ \ll\
\underbrace{p^{\text{open}}_\varphi(F_7)}_{\text{free generation}} ,
\tag{10}
$$

so that $\Delta_{F_7}$ is the interface-attributable component and
$p^{\text{clip}}_\varphi(F_7)$ the perception/label-hierarchy residual. Because only
$\mathcal{O}$ varies, $\Delta_{F_7}$ is causally attributable to the generation
interface (identification under: same $m$, image, family; $\mathrm{do}$ acts on
$\mathcal{O}$ alone).

## 8. Validity estimators (judge as a measurement instrument)

Let raters $r\in\{h_1,h_2,g\}$ (two humans, the judge) give labels $r(e)\in\mathcal{T}$
on a shared item set $S$. With observed and chance agreement
$p_o^{ab}=\tfrac{1}{|S|}\sum_{e}\mathbb{1}[a(e)=b(e)]$,
$p_e^{ab}=\sum_{F}\hat p_a(F)\hat p_b(F)$,

$$
\kappa_{ab}=\frac{p_o^{ab}-p_e^{ab}}{1-p_e^{ab}}
\quad(\text{Cohen}),
\qquad
\kappa^{\text{Fl}}(\mathcal{R})=\frac{\bar P-\bar P_e}{1-\bar P_e}
\quad(\text{Fleiss, }n=|\mathcal{R}|\text{ raters}).
\tag{11}
$$

**Envelope criterion (the validity notion).** The judge is *within the human
inter-annotator envelope* iff

$$
\min_i \kappa_{h_i g}\ \gtrsim\ \kappa_{h_1 h_2}
\quad\text{and}\quad
\big|\ \kappa^{\text{Fl}}(\{h_1,h_2,g\})-\kappa^{\text{Fl}}(\{h_1,h_2\})\ \big|\ \le\ \tau,
\tag{12}
$$

i.e. the judge agrees with humans about as well as humans agree with each other, and
is exchangeable with a human in the panel (small $\tau$). **Consensus accuracy** on
the agreement subset $C=\{e: h_1(e)=h_2(e)\}$,

$$
A_\varphi=\Pr\big[g(e)=h_1(e)\ \big|\ e\in C,\ \varphi(e)=\varphi\big],
\tag{13}
$$

isolates judge correctness where ground truth is least ambiguous.

## 9. Robustness and noise

**Inter-judge sensitivity.** For a panel $\{g^{(1)},\dots,g^{(8)}\}$ report
$\bar\kappa=\binom{8}{2}^{-1}\sum_{u<v}\kappa_{g^{(u)}g^{(v)}}$ and the dominant-mode
agreement $\mathbb{1}[\arg\max_F \hat p^{g^{(u)}}_\varphi(F)$ constant over $u]$; this
bounds the sensitivity of Eq. (3) to the choice of $g_\theta$.

**Gold-wrong / label noise.** Annotators may emit $\textsf{GW}\notin\mathcal{T}$ when
$y^\star$ is wrong and the model right; such items are removed from Eqs. (11)–(13) and
reported as a rate $\rho_{\textsf{GW}}=\Pr[r(e)=\textsf{GW}]$ (they are not model
failures).

---

## Assumptions (collected)

- **(A1) Judge consistency** — $g_\theta$ estimates $J$ up to noise validated by §8/§9.
- **(A2) Intervention exclusivity** — $\mathrm{do}(\mathcal{O})$ changes only the
  admissible output set (prompt-level label injection), so $\Delta_{F_7}$ is
  attributable to the interface.
- **(A3) Hierarchy well-definedness** — $\succ$ is available (or judge-inferred) so
  $F_6$ (lateral) vs $F_7$ (hypernym) is decidable.
- **(A4) Text-sufficiency** — $e'=(q,y^\star,\hat y,\varphi)$ carries enough signal
  for $J$; families where it fails (e.g. satellite land-use) are the disclosed
  low-$A_\varphi$ cases.

## Symbol table

| Symbol | Meaning | | Symbol | Meaning |
|---|---|---|---|---|
| $x,q,y^\star,\hat y$ | image, query, gold, prediction | | $\mathcal{T},F_k$ | taxonomy, mode $k$ |
| $\mathcal{L}_\mathcal{D}$ | dataset label set | | $J,g_\theta$ | diagnosis op., LLM judge |
| $\varphi,\mu$ | family, modality | | $R,\bot$ | matching cascade, no-match |
| $P^{\text{err}}_{m,\varphi}$ | error distribution | | $\mathcal{O}$ | admissible output set |
| $p_\varphi(F)$ | per-family failure dist. | | $\succ$ | hypernymy relation |
| $\Delta_{F_7},\Delta_{\text{acc}}$ | interface effects | | $\kappa,\kappa^{\text{Fl}}$ | Cohen / Fleiss agreement |
| $A_\varphi$ | consensus accuracy | | $\rho_{\textsf{GW}}$ | gold-wrong rate |
