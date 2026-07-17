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
                    use_embeddings: bool = False, label_hint_k: int = 0) -> dict:
    from datasets import load_dataset

    labels, synonyms = load_label_set(spec.label_set or "", repo_root)
    if not labels:
        raise RuntimeError(
            f"label set {spec.label_set!r} missing/empty; add "
            f"failuremodebench/labelsets/{spec.label_set}.json")
    matcher = LabelMatcher(labels, synonyms, use_embeddings=use_embeddings)
    # short prompt for large label spaces; enumerate options only when small
    hint = ", ".join(labels) if 0 < len(labels) <= (label_hint_k or 0) else ""
    prompt = classification_prompt(len(labels), hint)

    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.hf_split,
                      streaming=True) if spec.hf_config else \
        load_dataset(spec.hf_path, split=spec.hf_split, streaming=True)

    records: list[PredictionRecord] = []
    n = correct = kept = skipped_res = 0
    from .labelmatch import normalize
    gold_norm_lookup = {nl: i for i, nl in enumerate(matcher.norm_labels)}
    for ex in ds:
        image = ex.get("image") or ex.get("img") or ex.get("jpg")
        raw_label = ex.get(spec.label_field, ex.get("label"))
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
        ans = backend.generate(image, prompt)
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
