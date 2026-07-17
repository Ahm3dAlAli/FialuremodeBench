"""Unified per-sample prediction record + JSONL IO.

Both the recognition runner and the VQA (VLMEvalKit) importer emit the SAME
record shape, so error extraction, the judge and aggregation are modality-blind.
Images referenced by errors are written to <run>/images/ so the judge can
re-attach them.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PredictionRecord:
    sample_id: str
    dataset: str
    family: str
    modality: str          # recognition | vqa
    model: str
    question: str          # prompt / task text shown to the model
    gold: str              # gold answer / label (text)
    prediction: str        # raw model output text
    pred_label: str = ""   # normalised/extracted answer (recognition)
    correct: bool = False
    image_ref: str = ""    # path to saved image (for the judge), if any
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_jsonl(path: str, records) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write((r.to_json() if isinstance(r, PredictionRecord)
                     else json.dumps(r, ensure_ascii=False)) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_image(image, run_dir: str, sample_id: str) -> str:
    img_dir = os.path.join(run_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(sample_id))
    path = os.path.join(img_dir, f"{safe}.png")
    image.convert("RGB").save(path)
    return path
