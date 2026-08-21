# FailureModeBench — Representative Data Examples

What a data point looks like at each stage of the pipeline, and one representative error
per failure mode (drawn verbatim from `results/main2b/corpus.jsonl`).

---

## 1. The unified record (one row)

Every prediction — recognition or VQA — is stored in one modality-blind schema. Correct
predictions are set aside; only errors enter the judged corpus.

```jsonc
{
  "error_key":    "af31…d2",        // stable id = sha1(model | dataset | sample_id)
  "dataset":      "imagenet_r",
  "family":       "robustness",     // one of 9 task families
  "modality":     "recognition",    // recognition | vqa   (downstream is modality-blind)
  "model":        "qwen2vl_2b",
  "question":     "Identify the main subject of this image. Answer with the single most
                   specific category name only.",
  "gold":         "afghan_hound",   // y*  (dataset label)
  "prediction":   "Dog",            // ŷ   (raw model text — the judge sees THIS)
  "pred_label":   "",               // ŷ resolved to the label set (used for accuracy only)
  "correct":      false,            // ε = 1
  "image_ref":    "…/qwen2vl_2b-imagenet_r-0003912.png",   // saved only for errors
  // ---- added by the text-only LLM judge ----
  "failure_mode": "F7",             // one of F1–F8  (the estimand J(e))
  "confidence":   0.98,
  "secondary_mode":"F6",
  "rationale":    "Names the parent category 'Dog' instead of the breed 'afghan_hound'."
}
```

**Key point (reviewer W6):** the judge reads `prediction` (raw text `"Dog"`), **not**
`pred_label` (the resolved label). The matching cascade affects only `correct`/accuracy,
never the failure mode.

---

## 2. Pipeline flow for this example

```
 image  ─┐
         ├─►  Qwen2-VL-2B  ──►  ŷ = "Dog"        (free-form generation)
 "name  ─┘                        │
  the                             ├─►  cascade R:  "Dog" ∉ labels near "afghan_hound"
  class"                          │                 → correct = false  (an ERROR)
                                  │
                                  └─►  text-only judge g(q, y*="afghan_hound", ŷ="Dog")
                                        "Dog" ≻ "afghan_hound"  (hypernym)  →  F7
```

---

## 3. One representative error per failure mode

The two that carry the recognition analysis are **F7 vs F6** — *right idea too broad*
vs *wrong sibling*:

| Mode | Family / dataset | gold `y*` | prediction `ŷ` | why (judge rationale, abridged) |
|---|---|---|---|---|
| **F7** Overgeneralization | recognition / ImageNet-R | `afghan_hound` | **"Dog"** | names the **parent** category, not the breed (ŷ ≻ y*) |
| **F6** Misattribution | recognition / ImageNet-Sketch | `moving van` | **"Car"** | a **sibling** vehicle class, not a hypernym |
| **F1** Visual Hallucination | VQA / MathVista | `No` (people not in a circle) | **"Yes"** | asserts a spatial arrangement **not present** in the image |
| **F2** Linguistic Hallucination | VQA / CRPE | `book is over the table` | "book is *looking at* the table" | an **unsupported** verbal relation |
| **F3** Cross-Modal Misalign. | VQA / CRPE | `toilet is on the floor` | "the floor is on the floor" | a self-referential answer that **ignores the image** |
| **F4** Fusion Error | VQA / MathVerse | `7` | **"5"** | reads the coordinates right but **combines** them wrongly |
| **F5** Fabrication | VQA / MathVerse | `SA = 3298.67 cm²` | **"10.00"** | returns the inner radius — **invents** a value for the asked quantity |
| **F8** Confabulation | VQA / MME | `No` | "B" with a fluent but **false** justification | coherent reasoning, wrong conclusion |

**Food101 canonical pair (used in Table 1):** `beignets → "breakfast"` is **F7** (a correct
parent), whereas `beignets → "donuts"` is **F6** (a wrong look-alike sibling). Same image,
different failure — a distinction accuracy cannot make.

---

## 4. From records to the estimand

Aggregating `failure_mode` over the error set of a family gives the per-family distribution
$\hat p_\varphi(F)$ (Eq. 3). E.g. ImageNet-R recognition errors:

```
 fine_grained / ImageNet-R  (n = 896 errors)
   F7 Overgeneralization  ████████████████████████████████████████████  54%
   F6 Misattribution      ██████████                                     12%
   F1 Visual Halluc.      ████████                                       10%
   …
```

The paper reports one such distribution per family (Table 5) and the model × mode matrix
(Table 6); the closed-set control (Table 8) is the same $\hat p_\varphi(\text{F7})$
recomputed after constraining the output space.

---

*All examples are verbatim from `results/main2b/corpus.jsonl`; regenerate with the
`failuremodebench.judge.read_jsonl` utility or the scripts under `scripts/`.*
