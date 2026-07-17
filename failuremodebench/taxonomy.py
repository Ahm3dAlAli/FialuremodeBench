"""The 8-category failure taxonomy (paper §3.2) and the LLM-judge rubric.

Every *error* produced by a model under test (a sample it got wrong) is routed
to exactly one of the eight failure modes F1-F8 by an LLM judge. This module is
the single source of truth for the category definitions, the few-shot anchors,
and the judge prompt / output schema. `judge.py` imports from here so the rubric
and the parser can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureMode:
    code: str            # F1..F8
    name: str
    definition: str
    manifestation: str   # dataset-grounded example (paper table)


FAILURE_MODES: list[FailureMode] = [
    FailureMode(
        "F1", "Visual Hallucination",
        "Model reports visual entities or features that are absent from the image.",
        "ImageNet-A: 'sees' adversarial patterns that do not exist; "
        "RealWorldQA: reports non-existent objects."),
    FailureMode(
        "F2", "Linguistic Hallucination",
        "Generated text contains unsupported claims not grounded in the image or "
        "the question context.",
        "VQA responses include invented attributes; CRPE: adds details not in the caption."),
    FailureMode(
        "F3", "Cross-Modal Misalignment",
        "Visual and linguistic representations fail to align; the model ignores the "
        "image or mis-binds the text to it.",
        "Food101: describes a dish with wrong ingredient names despite the correct "
        "image; DTD: texture label mismatched."),
    FailureMode(
        "F4", "Fusion Error",
        "Incorrect integration of vision and language features during reasoning; the "
        "pieces are individually read correctly but combined wrongly.",
        "AI2D: misreads a chart axis; MathVista: combines numbers from the image incorrectly."),
    FailureMode(
        "F5", "Fabrication",
        "Inventing entire facts, steps or relationships with no basis in the input.",
        "MathVerse: invents a formula step; Capture: invents a causal chain."),
    FailureMode(
        "F6", "Misattribution",
        "Assigning properties, actions or categories to the wrong entity or region.",
        "ImageNet-R: assigns a texture to the wrong object class; SEED: misattributes "
        "an action to the wrong person/object."),
    FailureMode(
        "F7", "Overgeneralization",
        "Applying a broad / super-category label instead of the required fine-grained "
        "subclass.",
        "ImageNet: 'bird' instead of 'sparrow'; Resisc45: 'forest' instead of a "
        "specific land-use type."),
    FailureMode(
        "F8", "Confabulation",
        "Producing a coherent, plausible-sounding but false explanation or reasoning "
        "chain.",
        "MME: fluent but wrong explanation; CRPE: logical-sounding but inverted "
        "compositional relation."),
]

CODES = [m.code for m in FAILURE_MODES]
BY_CODE = {m.code: m for m in FAILURE_MODES}


# --------------------------------------------------------------------------- #
# Judge prompt                                                                 #
# --------------------------------------------------------------------------- #
def rubric_block() -> str:
    """The F1-F8 rubric, rendered for the judge system prompt."""
    lines = []
    for m in FAILURE_MODES:
        lines.append(f"{m.code} — {m.name}: {m.definition}\n    e.g. {m.manifestation}")
    return "\n".join(lines)


JUDGE_SYSTEM = f"""\
You are an expert annotator for FailureModeBench. A vision-language model was \
shown an image and a task, and produced an INCORRECT answer. Your job is to \
diagnose WHY it failed by assigning exactly ONE failure mode from this taxonomy:

{{rubric}}

Decision guidance (apply in this order):
1. If the model named something visible-but-wrong by using a broader parent \
category of the correct fine-grained label -> F7 Overgeneralization.
2. If the model referred to objects/attributes that are simply not in the image \
-> F1 Visual Hallucination (visual entity) or F2 Linguistic Hallucination \
(unsupported verbal claim).
3. If the model read the individual visual/textual facts right but combined them \
wrongly in a reasoning step -> F4 Fusion Error.
4. If the model invented whole steps, formulas or causal chains -> F5 Fabrication.
5. If it attached a real property/action to the wrong entity or region -> F6 \
Misattribution.
6. If the answer/rationale is fluent and internally coherent but the conclusion \
is false (and none of the above fits more precisely) -> F8 Confabulation.
7. If the output ignores the image or binds the text to the wrong modality \
-> F3 Cross-Modal Misalignment.

Pick the SINGLE best-fitting mode. Base the judgement only on the provided \
image, question, gold answer and model output. Do not reward or penalise \
verbosity. Return STRICT JSON and nothing else.""".format(rubric=rubric_block())


# JSON schema the judge must emit (also used to validate/parse the reply).
JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_mode": {"type": "string", "enum": CODES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "secondary_mode": {"type": "string", "enum": CODES + ["none"]},
    },
    "required": ["failure_mode", "confidence", "rationale"],
    "additionalProperties": False,
}


def user_prompt(question: str, gold: str, prediction: str,
                dataset: str, family: str) -> str:
    """The per-error user turn (image is attached separately by the provider)."""
    return (
        f"Dataset: {dataset}  (family: {family})\n"
        f"Task / question shown to the model:\n{question}\n\n"
        f"GOLD (correct) answer:\n{gold}\n\n"
        f"MODEL output (incorrect):\n{prediction}\n\n"
        "Diagnose the single failure mode (F1-F8). "
        'Respond with strict JSON: '
        '{"failure_mode":"F#","confidence":0-1,"rationale":"...",'
        '"secondary_mode":"F# or none"}'
    )
