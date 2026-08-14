"""Extract errors and route each to a failure mode F1-F8 with an LLM judge.

Pipeline:
  1. gather every incorrect PredictionRecord across the run;
  2. per task family, sample up to N errors (paper: 200/family), stratified by
     model+dataset so no single set dominates;
  3. for each sampled error, ask the judge (Claude by default) to assign one of
     F1-F8 from the taxonomy rubric, attaching the image when available;
  4. write an annotated error corpus (JSONL) that aggregation consumes.

The judge output is cached by a content hash and the corpus is written
incrementally, so a crashed/rate-limited run resumes without re-paying.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Optional

from . import taxonomy
from .config import DATASETS
from .records import read_jsonl


def gather_errors(run_dir: str) -> list[dict]:
    errors = []
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith(".jsonl") and "__" in fn and fn != "corpus.jsonl":
            for rec in read_jsonl(os.path.join(run_dir, fn)):
                if not rec.get("correct", False):
                    errors.append(rec)
    return errors


def sample_per_family(errors: list[dict], n_per_family: int,
                      seed: int = 0) -> list[dict]:
    """Deterministic stratified sample of <=n_per_family errors per family,
    spread across (model, dataset) cells. No RNG -> reproducible."""
    by_family: dict[str, list[dict]] = defaultdict(list)
    for e in errors:
        by_family[e["family"]].append(e)

    sampled = []
    for family, errs in sorted(by_family.items()):
        cells: dict[tuple, list[dict]] = defaultdict(list)
        for e in errs:
            cells[(e["model"], e["dataset"])].append(e)
        for c in cells:
            cells[c].sort(key=lambda e: e["sample_id"])
        picked, keys, idx = [], sorted(cells), 0
        # round-robin across cells for balance
        while len(picked) < min(n_per_family, len(errs)):
            k = keys[idx % len(keys)]
            if cells[k]:
                picked.append(cells[k].pop(0))
            idx += 1
            if idx > n_per_family * 8 and all(not cells[k] for k in keys):
                break
        sampled.extend(picked)
    return sampled


def _parse_verdict(text: str) -> Optional[dict]:
    """Extract the JSON verdict from the judge reply; tolerant of code fences."""
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1].lstrip("json").strip() if t.count("```") >= 2 else t
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        v = json.loads(t[start:end + 1])
    except Exception:
        return None
    fm = str(v.get("failure_mode", "")).upper()
    if fm not in taxonomy.CODES:
        return None
    v["failure_mode"] = fm
    return v


def _load_image(image_ref: str):
    if not image_ref or not os.path.exists(image_ref):
        return None
    try:
        from PIL import Image
        return Image.open(image_ref)
    except Exception:
        return None


def judge_errors(sampled: list[dict], provider, out_path: str,
                 attach_images: bool = True, max_retries: int = 3,
                 sleep_on_err: float = 2.0) -> dict:
    """Classify each sampled error; write annotated corpus incrementally."""
    done: dict[str, dict] = {}
    if os.path.exists(out_path):
        for r in read_jsonl(out_path):
            done[r["error_key"]] = r

    counts = defaultdict(int)
    fout = open(out_path, "a", encoding="utf-8")
    try:
        for e in sampled:
            ekey = hashlib.sha1(
                f"{e['model']}|{e['dataset']}|{e['sample_id']}".encode()).hexdigest()[:16]
            if ekey in done:
                counts[done[ekey].get("failure_mode", "NA")] += 1
                continue
            sys = taxonomy.JUDGE_SYSTEM
            usr = taxonomy.user_prompt(
                e.get("question", ""), e.get("gold", ""), e.get("prediction", ""),
                e["dataset"], e["family"])
            image = _load_image(e.get("image_ref", "")) if attach_images else None

            verdict = None
            for attempt in range(max_retries):
                try:
                    # Reasoning judges (e.g. deepseek-*) spend tokens on a hidden
                    # reasoning pass before emitting content; too small a budget
                    # truncates before any JSON and yields NA. 2048 leaves room.
                    reply = provider.complete(sys, usr, image=image,
                                              max_tokens=2048, temperature=0.0)
                    verdict = _parse_verdict(reply)
                    if verdict:
                        break
                except Exception as ex:  # rate limit / transient
                    if attempt == max_retries - 1:
                        verdict = {"failure_mode": "NA", "confidence": 0.0,
                                   "rationale": f"judge error: {ex}",
                                   "secondary_mode": "none"}
                    else:
                        time.sleep(sleep_on_err * (attempt + 1))
            if verdict is None:
                verdict = {"failure_mode": "NA", "confidence": 0.0,
                           "rationale": "unparseable judge reply",
                           "secondary_mode": "none"}

            row = {"error_key": ekey, "model": e["model"], "dataset": e["dataset"],
                   "family": e["family"], "modality": e["modality"],
                   "sample_id": e["sample_id"], "question": e.get("question", ""),
                   "gold": e.get("gold", ""), "prediction": e.get("prediction", ""),
                   "had_image": image is not None, **verdict}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            counts[verdict["failure_mode"]] += 1
    finally:
        fout.close()
    return dict(counts)
