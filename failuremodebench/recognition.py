"""Recognition (image-classification) runner.

For each recognition dataset (ImageNet family, DTD, Food101, Resisc45):
  1. stream the HF split, applying the >1000px resolution filter (paper §3.1);
  2. prompt the VLM to name the class;
  3. resolve the free-text answer to a canonical label (labelmatch);
  4. score top-1 accuracy (+ macro-F1 / confusion matrix for fine-grained sets);
  5. write unified PredictionRecords (only errors keep a saved image, to bound disk).

Runs on rolf via VLMEvalKitBackend; locally-testable via ApiVLMBackend.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .config import DATASETS, RESOLUTION_FILTER, DatasetSpec
from .labelmatch import LabelMatcher, classification_prompt
from .records import PredictionRecord, save_image, write_jsonl


def load_label_set(name: str, repo_root: str) -> tuple[list[str], dict]:
    """labelsets/<name>.json -> (labels, synonyms). Falls back to [] if absent."""
    path = os.path.join(repo_root, "failuremodebench", "labelsets", f"{name}.json")
    if not os.path.exists(path):
        return [], {}
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if isinstance(blob, list):
        return blob, {}
    return blob.get("labels", []), blob.get("synonyms", {})


def _oversized(image) -> bool:
    try:
        return max(image.size) > RESOLUTION_FILTER
    except Exception:
        return False


def run_recognition(spec: DatasetSpec, backend, model_key: str, run_dir: str,
                    repo_root: str, limit: Optional[int] = None,
                    use_embeddings: bool = False, label_hint_k: int = 0,
                    shuffle_labels: int = 0, hint_distractors: int = 0) -> dict:
    from datasets import load_dataset

    labels, synonyms = load_label_set(spec.label_set or "", repo_root)
    if not labels:
        raise RuntimeError(
            f"label set {spec.label_set!r} missing/empty; add "
            f"failuremodebench/labelsets/{spec.label_set}.json")
    matcher = LabelMatcher(labels, synonyms, use_embeddings=use_embeddings)
    # closed-set control: enumerate candidate labels in the prompt when the label
    # space is small enough. shuffle_labels>0 permutes the order (ordering-bias
    # robustness check, reviewer W2/Q3); hint_distractors>0 enables the large-label
    # variant (gold + K random distractors per item, reviewer W4/Q5 -- handled in
    # the per-example loop below rather than a fixed global hint).
    import random as _random
    _hint_labels = list(labels)
    if shuffle_labels:
        _random.Random(shuffle_labels).shuffle(_hint_labels)
    hint = ", ".join(_hint_labels) if (0 < len(labels) <= (label_hint_k or 0) and not hint_distractors) else ""
    prompt = classification_prompt(len(labels), hint)

    # split arithmetic ("train+val+test") is not supported in streaming mode,
    # so stream single splits (large ImageNet sets) and load concatenated splits
    # eagerly (only used for the small DTD set).
    if spec.local_path:
        # local ImageFolder mirror (e.g. shared ImageNet val); label = sorted
        # folder index, which for wnid dirs == the standard ImageNet order.
        # imagefolder names the split after the dir (val -> 'validation'), so
        # take whatever single split it produces.
        dd = load_dataset("imagefolder", data_dir=spec.local_path, streaming=True)
        ds = dd[next(iter(dd.keys()))]
    else:
        # Datasets are prefetched to the local HF cache, so load NON-streaming
        # (reads from cache; avoids the HF streaming deadlock observed on rolf,
        # which hung the 7B on the first streaming split).
        stream = False
        _kw = dict(split=spec.hf_split, streaming=stream)
        if spec.trust_remote_code:
            _kw["trust_remote_code"] = True
        if not stream:
            _kw["verification_mode"] = "no_checks"  # tolerate split drift (ImageNet-V2)
        ds = load_dataset(spec.hf_path, spec.hf_config, **_kw) if spec.hf_config else \
            load_dataset(spec.hf_path, **_kw)
    per_class_cap = spec.subsample_per_class      # 0 = keep all
    per_class_seen: dict = {}

    records: list[PredictionRecord] = []
    n = correct = kept = skipped_res = 0
    from .labelmatch import normalize
    gold_norm_lookup = {nl: i for i, nl in enumerate(matcher.norm_labels)}
    for ex in ds:
        image = (ex.get("image") or ex.get("img") or ex.get("jpg")
                 or ex.get("jpeg") or ex.get("png"))
        raw_label = ex.get(spec.label_field, ex.get("label"))
        if raw_label is None and ex.get("__key__"):
            # webdataset (e.g. vaishaal/ImageNetV2): class index is the leading
            # integer of the __key__ path (e.g. "281/281_5" -> 281).
            for part in str(ex["__key__"]).replace("_", "/").split("/"):
                if part.isdigit():
                    raw_label = int(part)
                    break
        if image is None or raw_label is None:
            continue
        if _oversized(image):
            skipped_res += 1
            continue
        kept += 1
        if spec.label_is_index and isinstance(raw_label, int):
            gold = labels[raw_label] if raw_label < len(labels) else str(raw_label)
            gold_idx = raw_label
        else:  # string label -> resolve to a label-set index for scoring
            gold = str(raw_label)
            gold_idx = gold_norm_lookup.get(normalize(gold))
        if per_class_cap:  # balanced N/class subsample (deterministic by stream order)
            key = gold_idx if gold_idx is not None else gold
            if per_class_seen.get(key, 0) >= per_class_cap:
                continue
            per_class_seen[key] = per_class_seen.get(key, 0) + 1
        _prompt = prompt
        if hint_distractors:  # large-label closed-set: gold + K random distractors
            pool = [l for l in labels if normalize(l) != normalize(gold)]
            seed = (shuffle_labels or 1) * 1000003 + n
            cand = _random.Random(seed).sample(pool, min(hint_distractors, len(pool))) + [gold]
            _random.Random(seed + 1).shuffle(cand)
            _prompt = classification_prompt(len(labels), ", ".join(cand))
        ans = backend.generate(image, _prompt)
        mi, method = matcher.match(ans)
        pred_label = labels[mi] if mi is not None else ""
        if gold_idx is not None:
            is_correct = (mi is not None and
                          matcher.norm_labels[mi] == matcher.norm_labels[gold_idx])
        else:
            is_correct = pred_label.lower() == gold.lower()
        n += 1
        correct += int(is_correct)
        sid = f"{spec.key}-{n:07d}"
        img_ref = "" if is_correct else save_image(image, run_dir, f"{model_key}-{sid}")
        records.append(PredictionRecord(
            sample_id=sid, dataset=spec.key, family=spec.family,
            modality="recognition", model=model_key, question=prompt,
            gold=gold, prediction=ans, pred_label=pred_label,
            correct=is_correct, image_ref=img_ref,
            extra={"match_method": method}))
        if limit and n >= limit:
            break

    out = os.path.join(run_dir, f"{model_key}__{spec.key}.jsonl")
    write_jsonl(out, records)
    acc = correct / n if n else 0.0
    return {"dataset": spec.key, "model": model_key, "n": n, "correct": correct,
            "accuracy": acc, "skipped_resolution": skipped_res,
            "predictions": out}


def run_recognition_clip(spec: DatasetSpec, backend, model_key: str, run_dir: str,
                         repo_root: str, limit: Optional[int] = None) -> dict:
    """CLIP zero-shot classification over the dataset label set. Same record schema
    as run_recognition (so judging/aggregation are identical), but prediction is the
    argmax label from the fixed set -- CLIP cannot emit an out-of-set super-category.
    """
    from datasets import load_dataset
    from .labelmatch import normalize

    labels, synonyms = load_label_set(spec.label_set or "", repo_root)
    if not labels:
        raise RuntimeError(f"label set {spec.label_set!r} missing/empty")
    backend.set_labels(labels)                      # precompute text embeddings once
    norm_labels = [normalize(l) for l in labels]
    gold_norm_lookup = {nl: i for i, nl in enumerate(norm_labels)}

    if spec.local_path:
        dd = load_dataset("imagefolder", data_dir=spec.local_path, streaming=True)
        ds = dd[next(iter(dd.keys()))]
    else:
        _kw = dict(split=spec.hf_split, streaming=False, verification_mode="no_checks")
        if spec.trust_remote_code:
            _kw["trust_remote_code"] = True
        ds = load_dataset(spec.hf_path, spec.hf_config, **_kw) if spec.hf_config else \
            load_dataset(spec.hf_path, **_kw)

    per_class_cap = spec.subsample_per_class
    per_class_seen: dict = {}
    records: list[PredictionRecord] = []
    n = correct = kept = skipped_res = 0
    for ex in ds:
        image = (ex.get("image") or ex.get("img") or ex.get("jpg")
                 or ex.get("jpeg") or ex.get("png"))
        raw_label = ex.get(spec.label_field, ex.get("label"))
        if raw_label is None and ex.get("__key__"):
            for part in str(ex["__key__"]).replace("_", "/").split("/"):
                if part.isdigit():
                    raw_label = int(part); break
        if image is None or raw_label is None:
            continue
        if _oversized(image):
            skipped_res += 1; continue
        kept += 1
        if spec.label_is_index and isinstance(raw_label, int):
            gold = labels[raw_label] if raw_label < len(labels) else str(raw_label)
            gold_idx = raw_label
        else:
            gold = str(raw_label); gold_idx = gold_norm_lookup.get(normalize(gold))
        if per_class_cap:
            key = gold_idx if gold_idx is not None else gold
            if per_class_seen.get(key, 0) >= per_class_cap:
                continue
            per_class_seen[key] = per_class_seen.get(key, 0) + 1
        pred_label, score = backend.classify(image)
        pred_idx = gold_norm_lookup.get(normalize(pred_label))
        if gold_idx is not None and pred_idx is not None:
            is_correct = (pred_idx == gold_idx)
        else:
            is_correct = pred_label.lower() == gold.lower()
        n += 1
        correct += int(is_correct)
        sid = f"{spec.key}-{n:07d}"
        img_ref = "" if is_correct else save_image(image, run_dir, f"{model_key}-{sid}")
        records.append(PredictionRecord(
            sample_id=sid, dataset=spec.key, family=spec.family,
            modality="recognition", model=model_key, question="[CLIP zero-shot]",
            gold=gold, prediction=pred_label, pred_label=pred_label,
            correct=is_correct, image_ref=img_ref,
            extra={"match_method": "clip_zeroshot", "score": round(score, 4)}))
        if limit and n >= limit:
            break

    out = os.path.join(run_dir, f"{model_key}__{spec.key}.jsonl")
    write_jsonl(out, records)
    return {"dataset": spec.key, "model": model_key, "n": n, "correct": correct,
            "accuracy": correct / n if n else 0.0, "skipped_resolution": skipped_res,
            "predictions": out}
