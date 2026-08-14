# Human annotations

Blind expert annotations of model errors with the F1–F8 failure taxonomy, used to
validate the LLM judge (`§Judge Reliability` in the paper).

| File | Annotator | Items labeled |
|---|---|---|
| `annotator1_ahmed.json` | Annotator 1 | 700-item study |
| `annotator2_cui.json` | Annotator 2 (Cui) | 648/700 |
| `_archive/` | earlier / partial passes | (not used) |

**Format.** A JSON list; each row: `error_key` (join key to `results/main2b/corpus.jsonl`),
`family`, `judge` (the LLM-judge label at export time), and `human` — the annotator's
F1–F8 label, or `GW` = "gold wrong / model actually correct" (label noise; excluded
from κ). Rows with `human: null` were left unlabeled.

**Reproduce the judge-validity analysis** (per-annotator vs judge, human–human κ,
Fleiss' κ, judge-vs-consensus):

    python scripts/multi_annotator_agreement.py --run main2b

**Add a new annotator:** drop `annotatorN_name.json` (exported from the study page
`docs/FailureModeBench_annotation_study.html`) into this folder and re-run the command
above — it auto-discovers every `*.json` here.
