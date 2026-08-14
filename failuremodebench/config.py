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
# Nine task families (paper §Benchmark / Table 4). CRPE (compositional
# relations + the hallucinations they induce) is folded into "general_vqa",
# matching the paper's "General (comp.)" grouping — so there are 9, not 10.
TASK_FAMILIES = [
    "generic",          # ImageNet, ImageNet-V2
    "robustness",       # ImageNet-A/R/Sketch, DTD
    "fine_grained",     # Food101
    "satellite",        # Resisc45
    "math",             # MathVerse, MathVista
    "general_vqa",      # SEED, MME, CRPE
    "real_world",       # RealWorldQA
    "counterfactual",   # Capture
    "chart",            # AI2D
]

# Display labels for tables/figures (paper Table 4 row names).
FAMILY_DISPLAY = {
    "generic": "Generic", "robustness": "Robustness", "fine_grained": "Fine-grained",
    "satellite": "Satellite", "math": "Math", "general_vqa": "General",
    "real_world": "Real-World", "counterfactual": "Counterfact.", "chart": "Chart",
}


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
    trust_remote_code: bool = False    # HF repos with custom loading scripts
    subsample_per_class: int = 0       # keep only first N imgs/class (0 = all)
    local_path: Optional[str] = None   # load via imagefolder from this dir (skip HF)
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
        label_set="imagenet", metric="accuracy",
        # reuse the shared on-disk mirror (ImageFolder, sorted wnid == std order);
        # avoids the gated HF download + login
        local_path="/local/scratch/datasets/ImageNet/ILSVRC2012/val"),
    "imagenet_v2": DatasetSpec(
        "imagenet_v2", "generic", "recognition", "Generic", 9772, 10000,
        hf_path="vaishaal/ImageNetV2", hf_config=None,
        # default 'train' concatenates 3 variants (30k); first 10k = matched-frequency
        hf_split="train[:10000]", label_set="imagenet", metric="accuracy"),
    "imagenet_a": DatasetSpec(
        "imagenet_a", "robustness", "recognition", "Generic (adv.)", 7467, 7500,
        hf_path="barkermrl/imagenet-a", hf_split="train",  # repo exposes the 7.5k set as 'train'
        label_set="imagenet_a", metric="accuracy"),
    "imagenet_r": DatasetSpec(
        "imagenet_r", "robustness", "recognition", "Texture", 28506, 30000,
        hf_path="axiong/imagenet-r", hf_split="test",
        label_set="imagenet_r", metric="accuracy",
        label_field="class_name", label_is_index=False),
    "imagenet_sketch": DatasetSpec(
        "imagenet_sketch", "robustness", "recognition", "Edges", 35350, 50000,
        hf_path="vaughankraska/imagenet_sketch", hf_split="train+validation",  # full 50889; canonical repo deleted
        label_set="imagenet", metric="accuracy"),
    "dtd": DatasetSpec(
        "dtd", "robustness", "recognition", "Edges, Texture", 5640, 5640,
        hf_path="tanganke/dtd", hf_split="train+test",  # only train(3760)+test(1880) = 5640
        label_set="dtd", metric="f1", fine_grained=True),
    # ---- Recognition: fine-grained + satellite -----------------------------
    "food101": DatasetSpec(
        "food101", "fine_grained", "recognition", "Fine-grained", 25250, 25250,
        hf_path="ethz/food101", hf_split="validation",   # namespaced; bare 'food101' rejected by hub
        label_set="food101", metric="f1", fine_grained=True),
    "resisc45": DatasetSpec(
        "resisc45", "satellite", "recognition", "Satellite Imagery", 4500, 4500,
        hf_path="tanganke/resisc45", hf_split="test",   # test=6300 (140/class)
        label_set="resisc45", metric="f1", fine_grained=True,
        subsample_per_class=100),   # -> 45x100 = 4500 to match paper Table 2
    # ---- VQA: math ---------------------------------------------------------
    "mathverse": DatasetSpec(
        "mathverse", "math", "vqa", "Mathematical Ability", 1631, 2180,
        vlmevalkit_name="MathVerse_MINI", metric="vlmevalkit"),
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
    "naturalbench": DatasetSpec(
        "naturalbench", "counterfactual", "vqa", "Counterfactual (NaturalBench)", 1900, 1900,
        vlmevalkit_name="NaturalBenchDataset", metric="vlmevalkit"),  # real counterfactual substitute for Capture
    "crpe": DatasetSpec(
        "crpe", "general_vqa", "vqa", "General (comp.)", 7575, 7575,
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
    "qwen2vl_2b": ModelSpec(
        "qwen2vl_2b", "Qwen2-VL-2B", "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen2-VL-2B-Instruct", "qwen2vl", True, load_4bit=True,
        notes="small; fits a contended 11GB GPU with no offload (interim tier)"),
    "qwen25vl_3b": ModelSpec(
        "qwen25vl_3b", "Qwen2.5-VL-3B", "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-3B-Instruct", "qwen2vl", True, load_4bit=True,
        notes="small; uses the patched Qwen2VL class; 2nd model-under-test"),
    "internvl25_2b": ModelSpec(
        "internvl25_2b", "InternVL2.5-2B", "OpenGVLab/InternVL2_5-2B",
        "InternVL2_5-2B", "internvl", True, load_4bit=True,
        notes="small InternVL; different arch family for cross-arch comparison"),
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
    # Contrastive (CLIP-style) recognisers -- zero-shot classification over the
    # dataset label set via open_clip. Recognition-only. hf_id encodes the
    # open_clip "arch:pretrained" spec. These are the newest-generation encoders
    # and serve as the architectural CONTRAST to generative VLMs: constrained to
    # the label set, they structurally cannot overgeneralise (F7).
    "siglip2_so400m": ModelSpec(
        "siglip2_so400m", "SigLIP2-SO400M", "ViT-SO400M-16-SigLIP2-384:webli",
        None, "clip", True, notes="SigLIP2 SO400M/384 (2025); open_clip zero-shot"),
    "metaclip2_h14": ModelSpec(
        "metaclip2_h14", "MetaCLIP2-H14", "ViT-H-14-quickgelu:metaclip_fullcc",
        None, "clip", True, notes="MetaCLIP (H/14 fullcc); open_clip zero-shot"),
    "dfn_h14_378": ModelSpec(
        "dfn_h14_378", "DFN-CLIP-H14-378", "ViT-H-14-378-quickgelu:dfn5b",
        None, "clip", True, notes="DFN-5B CLIP H/14 @378 (Apple); open_clip zero-shot"),
}

# open_clip fallbacks if a primary arch:pretrained tag is unavailable in the
# installed open_clip version (verified/adjusted at deploy time on rolf).
CLIP_FALLBACKS = {
    "siglip2_so400m": ["ViT-SO400M-14-SigLIP-384:webli", "ViT-L-16-SigLIP-384:webli"],
    "metaclip2_h14": ["ViT-H-14-quickgelu:metaclip_fullcc", "ViT-H-14:metaclip_fullcc"],
    "dfn_h14_378": ["ViT-H-14-quickgelu:dfn5b", "ViT-H-14:laion2b_s32b_b79k"],
}

# The default evaluation matrix (paper §3.3). Adjust in configs/matrix.yaml.
DEFAULT_MODELS = ["qwen2vl_7b", "internvl25_8b", "llava16_7b"]
DEFAULT_DATASETS = list(DATASETS.keys())

RESOLUTION_FILTER = 1000  # drop images with any side > this (paper §3.1)


def datasets_by_modality(modality: str) -> list[DatasetSpec]:
    return [d for d in DATASETS.values() if d.modality == modality]


def datasets_by_family(family: str) -> list[DatasetSpec]:
    return [d for d in DATASETS.values() if d.family == family]
