"""Central registries for FailureModeBench.

Everything downstream (runners, error extraction, the failure-mode judge,
aggregation) keys off the three registries defined here:

  * DATASETS  -- the 16 benchmark datasets, grouped into task families.
  * MODELS    -- the vision-language models under test.
  * TASK_FAMILIES -- the coarse grouping used for per-family reporting.

The 16 datasets split into two evaluation *modalities*:

  * "recognition"  -- image classification. The VLM is shown an image and asked
                      to name the class; the answer is matched against a fixed
                      label set. Scored with top-1 accuracy (+ per-class F1 /
                      confusion matrix for the fine-grained sets).
  * "vqa"          -- (multiple-choice or open) visual question answering. Run
                      through VLMEvalKit, which owns the loaders, prompt
                      templates and answer-extraction for these datasets.

The `Used Test Size` / `Original Size` columns from the paper's Table 3.1 are
recorded so the harness can (a) apply the >1000x1000 resolution filter and
(b) assert that the post-filter counts match what the paper reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Task families (paper §3.1)                                                   #
# --------------------------------------------------------------------------- #
TASK_FAMILIES = [
    "generic",          # ImageNet, ImageNet-V2
    "robustness",       # ImageNet-A/R/Sketch, DTD
    "fine_grained",     # Food101
    "satellite",        # Resisc45
    "math",             # MathVerse, MathVista
    "general_vqa",      # SEED, MME
    "real_world",       # RealWorldQA
    "counterfactual",   # Capture
    "compositional",    # CRPE
    "chart",            # AI2D
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str                       # our canonical id
    family: str                    # one of TASK_FAMILIES
    modality: str                  # "recognition" | "vqa"
    focus: str                     # human-readable focus (paper column)
    used_size: int                 # post-filter test count (paper)
    original_size: int             # original test count (paper)
    # recognition-only:
    hf_path: Optional[str] = None      # HuggingFace datasets path
    hf_split: str = "test"
    hf_config: Optional[str] = None
    label_set: Optional[str] = None    # name of label list in labelsets/
    label_field: str = "label"         # HF column holding the gold label
    label_is_index: bool = True        # True: int index into label_set; False: string
    # vqa-only:
    vlmevalkit_name: Optional[str] = None   # dataset id inside VLMEvalKit
    # scoring
    metric: str = "accuracy"           # "accuracy" | "f1" | "vlmevalkit"
    fine_grained: bool = False         # emit a confusion matrix


# --------------------------------------------------------------------------- #
# The 16 datasets (paper Table, §3.1). used/original sizes are the paper's.    #
# --------------------------------------------------------------------------- #
DATASETS: dict[str, DatasetSpec] = {
    # ---- Recognition: generic + robustness ---------------------------------
    "imagenet": DatasetSpec(
        "imagenet", "generic", "recognition", "Generic", 49032, 50000,
        hf_path="imagenet-1k", hf_split="validation",
        label_set="imagenet", metric="accuracy"),
    "imagenet_v2": DatasetSpec(
        "imagenet_v2", "generic", "recognition", "Generic", 9772, 10000,
        hf_path="vaishaal/ImageNetV2", hf_config="matched-frequency",
        hf_split="test", label_set="imagenet", metric="accuracy"),
    "imagenet_a": DatasetSpec(
        "imagenet_a", "robustness", "recognition", "Generic (adv.)", 7467, 7500,
        hf_path="barkermrl/imagenet-a", hf_split="test",
        label_set="imagenet_a", metric="accuracy"),
    "imagenet_r": DatasetSpec(
        "imagenet_r", "robustness", "recognition", "Texture", 28506, 30000,
        hf_path="axiong/imagenet-r", hf_split="test",
        label_set="imagenet_r", metric="accuracy",
        label_field="class_name", label_is_index=False),
    "imagenet_sketch": DatasetSpec(
        "imagenet_sketch", "robustness", "recognition", "Edges", 35350, 50000,
        hf_path="imagenet_sketch", hf_split="test",
        label_set="imagenet", metric="accuracy"),
    "dtd": DatasetSpec(
        "dtd", "robustness", "recognition", "Edges, Texture", 5640, 5640,
        hf_path="tanganke/dtd", hf_split="test",
        label_set="dtd", metric="f1", fine_grained=True),
    # ---- Recognition: fine-grained + satellite -----------------------------
    "food101": DatasetSpec(
        "food101", "fine_grained", "recognition", "Fine-grained", 25250, 25250,
        hf_path="food101", hf_split="validation",
        label_set="food101", metric="f1", fine_grained=True),
    "resisc45": DatasetSpec(
        "resisc45", "satellite", "recognition", "Satellite Imagery", 4500, 4500,
        hf_path="tanganke/resisc45", hf_split="test",
        label_set="resisc45", metric="f1", fine_grained=True),
    # ---- VQA: math ---------------------------------------------------------
    "mathverse": DatasetSpec(
        "mathverse", "math", "vqa", "Mathematical Ability", 1631, 2180,
        vlmevalkit_name="MathVerse_MINI_Vision", metric="vlmevalkit"),
    "mathvista": DatasetSpec(
        "mathvista", "math", "vqa", "Mathematical Ability", 490, 1000,
        vlmevalkit_name="MathVista_MINI", metric="vlmevalkit"),
    # ---- VQA: general understanding ---------------------------------------
    "seed": DatasetSpec(
        "seed", "general_vqa", "vqa", "General Understanding", 3881, 13991,
        vlmevalkit_name="SEEDBench_IMG", metric="vlmevalkit"),
    "mme": DatasetSpec(
        "mme", "general_vqa", "vqa", "General Understanding", 1576, 2370,
        vlmevalkit_name="MME", metric="vlmevalkit"),
    # ---- VQA: real-world / counterfactual / compositional / chart ---------
    "realworldqa": DatasetSpec(
        "realworldqa", "real_world", "vqa", "Realworld Understanding", 765, 765,
        vlmevalkit_name="RealWorldQA", metric="vlmevalkit"),
    "capture": DatasetSpec(
        "capture", "counterfactual", "vqa", "Counterfactual Understanding", 817, 962,
        vlmevalkit_name="COCO_VAL", metric="vlmevalkit"),   # placeholder id; see notes
    "crpe": DatasetSpec(
        "crpe", "compositional", "vqa", "Compositionality & Hallucination", 7575, 7575,
        vlmevalkit_name="CRPE_RELATION", metric="vlmevalkit"),
    "ai2d": DatasetSpec(
        "ai2d", "chart", "vqa", "Graph & Chart Understanding", 2704, 3090,
        vlmevalkit_name="AI2D_TEST", metric="vlmevalkit"),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    hf_id: Optional[str]           # HuggingFace weights (local/rolf backend)
    vlmevalkit_name: Optional[str] # id inside VLMEvalKit
    family: str                    # arch family for grouping
    is_open: bool
    load_4bit: bool = False        # quantise to fit an 11GB 2080 Ti
    notes: str = ""


# The paper's four "open and closed" VLMs (+ size variants it lists).
MODELS: dict[str, ModelSpec] = {
    "qwen2vl_7b": ModelSpec(
        "qwen2vl_7b", "Qwen2-VL-7B", "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen2-VL-7B-Instruct", "qwen2vl", True, load_4bit=True,
        notes="dynamic resolution; cap max pixels on 11GB cards"),
    "internvl25_4b": ModelSpec(
        "internvl25_4b", "InternVL2.5-4B", "OpenGVLab/InternVL2_5-4B",
        "InternVL2_5-4B", "internvl", True, load_4bit=False),
    "internvl25_8b": ModelSpec(
        "internvl25_8b", "InternVL2.5-8B", "OpenGVLab/InternVL2_5-8B",
        "InternVL2_5-8B", "internvl", True, load_4bit=True),
    "llava16_7b": ModelSpec(
        "llava16_7b", "LLaVA-1.6-7B", "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava_next_vicuna_7b", "llava", True, load_4bit=True),
    "llava16_13b": ModelSpec(
        "llava16_13b", "LLaVA-1.6-13B", "llava-hf/llava-v1.6-vicuna-13b-hf",
        "llava_next_vicuna_13b", "llava", True, load_4bit=True),
    # Closed / API model-under-test AND default judge backend.
    "claude_opus": ModelSpec(
        "claude_opus", "Claude-Opus-4.8", None, None, "api", False,
        notes="API VLM; also usable as the failure-mode judge"),
}

# The default evaluation matrix (paper §3.3). Adjust in configs/matrix.yaml.
DEFAULT_MODELS = ["qwen2vl_7b", "internvl25_8b", "llava16_7b"]
DEFAULT_DATASETS = list(DATASETS.keys())

RESOLUTION_FILTER = 1000  # drop images with any side > this (paper §3.1)


def datasets_by_modality(modality: str) -> list[DatasetSpec]:
    return [d for d in DATASETS.values() if d.modality == modality]


def datasets_by_family(family: str) -> list[DatasetSpec]:
    return [d for d in DATASETS.values() if d.family == family]
