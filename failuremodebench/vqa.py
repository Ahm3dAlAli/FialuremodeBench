"""VQA runner: drive VLMEvalKit, then import its predictions into our schema.

VLMEvalKit owns the loaders, prompt templates and (crucially) the answer
extraction / judging for MathVerse, MathVista, SEED, MME, RealWorldQA, CRPE,
AI2D and Capture. We shell out to its `run` entry-point on rolf, then parse the
per-sample prediction spreadsheets it writes into unified PredictionRecords so
the failure-mode judge and aggregation treat VQA and recognition identically.

VLMEvalKit column names drift across versions and datasets, so the importer maps
a set of likely aliases and records the correctness column it found.
"""
from __future__ import annotations

import base64
import io
import os
import subprocess
from typing import Optional

from .config import DATASETS, MODELS, DatasetSpec
from .records import PredictionRecord, write_jsonl


_Q_COLS = ["question", "query", "prompt", "Question"]
_GOLD_COLS = ["answer", "gt", "GT", "gold", "target", "Answer"]
_PRED_COLS = ["prediction", "pred", "response", "output", "Prediction"]
_HIT_COLS = ["hit", "score", "correct", "match", "Hit", "Score"]
_IMG_COLS = ["image", "image_path", "img", "Image"]


def run_vlmevalkit(model_key: str, spec: DatasetSpec, work_dir: str,
                   extra_args: Optional[list[str]] = None) -> str:
    """Invoke VLMEvalKit for one (model, dataset). Returns its output dir.

    Requires VLMEvalKit installed and the model/judge env configured (rolf).
    """
    spec_model = MODELS[model_key]
    vk_model = spec_model.vlmevalkit_name
    if not vk_model:
        raise ValueError(f"{model_key} has no vlmevalkit_name (API-only model).")
    out_dir = os.path.join(work_dir, "vlmevalkit", model_key)
    os.makedirs(out_dir, exist_ok=True)
    # VLMEvalKit's entry point is run.py at its repo root (no vlmeval.run module).
    import vlmeval
    run_py = os.path.join(os.path.dirname(os.path.dirname(vlmeval.__file__)), "run.py")
    launcher = [run_py] if os.path.exists(run_py) else ["-m", "vlmeval.run"]
    # --mode infer: inference only. Skips VLMEvalKit's answer-extraction step,
    # which for many datasets calls an OpenAI judge (gpt-4o-mini) and would hang
    # without a key. We score correctness ourselves in import_predictions.
    cmd = ["python", *launcher,
           "--model", vk_model,
           "--data", spec.vlmevalkit_name,
           "--mode", "infer",
           "--work-dir", out_dir]
    if extra_args:
        cmd += extra_args
    env = dict(os.environ)
    print("[vqa] $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    return out_dir


def _find_pred_file(out_dir: str, spec: DatasetSpec) -> Optional[str]:
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if fn.endswith((".xlsx", ".tsv", ".csv")) and spec.vlmevalkit_name in fn:
                # prefer the evaluated/scored file if present
                if "acc" not in fn.lower():
                    return os.path.join(root, fn)
    # fall back to any matching spreadsheet
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if fn.endswith((".xlsx", ".tsv", ".csv")) and spec.vlmevalkit_name in fn:
                return os.path.join(root, fn)
    return None


def _pick(row, cols, default=""):
    for c in cols:
        if c in row and row[c] not in (None, "") and str(row[c]) != "nan":
            return row[c]
    return default


import re as _re


def _vqa_correct(gold: str, pred: str, row: dict) -> bool:
    """Score a VQA prediction when VLMEvalKit didn't (mode=infer).
    Handles multiple-choice (gold is a letter / choice text) and open answers."""
    g = str(gold).strip()
    p = str(pred).strip()
    if not g:
        return False
    gl, pl = g.lower(), p.lower()
    # multiple-choice: gold is a single option letter A-E
    if _re.fullmatch(r"[A-Ea-e]", g):
        # predicted letter: leading "A", "A.", "(A)", "answer is A", or bare letter
        m = _re.search(r"\b([A-E])\b", p.upper())
        if m and m.group(1) == g.upper():
            return True
        # match against the gold option's text (row['A'..'E'])
        opt = row.get(g.upper())
        if opt and str(opt).strip() and str(opt).strip().lower() in pl:
            return True
        return False
    # yes/no
    if gl in ("yes", "no"):
        return _re.search(rf"\b{gl}\b", pl) is not None
    # open / numeric: exact, containment, or numeric equality
    if gl == pl or gl in pl or pl in gl:
        return True
    gn = _re.findall(r"-?\d+\.?\d*", g)
    pn = _re.findall(r"-?\d+\.?\d*", p)
    if gn and pn:
        try:
            return abs(float(gn[0]) - float(pn[0])) < 1e-6
        except ValueError:
            pass
    return False


def _decode_correct(val) -> Optional[bool]:
    if val in ("", None):
        return None
    s = str(val).strip().lower()
    if s in ("1", "1.0", "true", "yes", "hit", "correct"):
        return True
    if s in ("0", "0.0", "false", "no", "miss", "wrong", "incorrect"):
        return False
    try:
        return float(s) >= 0.5
    except ValueError:
        return None


def import_predictions(pred_file: str, model_key: str, spec: DatasetSpec,
                       run_dir: str, image_out: bool = True) -> tuple[str, dict]:
    """Parse a VLMEvalKit prediction spreadsheet -> unified JSONL."""
    import pandas as pd
    df = (pd.read_excel(pred_file) if pred_file.endswith(".xlsx")
          else pd.read_csv(pred_file, sep="\t" if pred_file.endswith(".tsv") else ","))
    cols = list(df.columns)
    hit_col = next((c for c in _HIT_COLS if c in cols), None)

    records, n, correct, unknown = [], 0, 0, 0
    img_dir = os.path.join(run_dir, "images")
    if image_out:
        os.makedirs(img_dir, exist_ok=True)
    for i, row in df.iterrows():
        row = row.to_dict()
        q = str(_pick(row, _Q_COLS))
        gold = str(_pick(row, _GOLD_COLS))
        pred = str(_pick(row, _PRED_COLS))
        corr = _decode_correct(row.get(hit_col)) if hit_col else None
        if corr is None:
            corr = _vqa_correct(gold, pred, row)
            unknown += 1
        sid = f"{spec.key}-{row.get('index', i)}"
        img_ref = ""
        raw_img = _pick(row, _IMG_COLS)
        if image_out and raw_img and isinstance(raw_img, str) and len(raw_img) > 200:
            try:  # base64-embedded image
                img = base64.b64decode(raw_img)
                img_ref = os.path.join(img_dir, f"{model_key}-{sid}.png")
                with open(img_ref, "wb") as fh:
                    fh.write(img)
            except Exception:
                img_ref = ""
        elif raw_img and os.path.exists(str(raw_img)):
            img_ref = str(raw_img)
        n += 1
        correct += int(bool(corr))
        # Multiple-choice: a bare-letter gold ("C") is meaningless to the text-only
        # judge and produces unclassifiable (NA) verdicts. Resolve the letter to its
        # option text and stash all options so the judge sees the actual answer.
        extra = {"hit_col": hit_col or "inferred"}
        gold_out = gold
        opts = {L: str(row[L]) for L in ["A", "B", "C", "D", "E"]
                if L in row and str(row.get(L, "nan")) != "nan"}
        if opts and _re.fullmatch(r"[A-Ea-e]", gold.strip()):
            gtext = opts.get(gold.strip().upper(), "")
            if gtext:
                gold_out = f"{gold} ({gtext})"
            extra["options"] = opts
            extra["gold_text"] = gtext
        records.append(PredictionRecord(
            sample_id=sid, dataset=spec.key, family=spec.family, modality="vqa",
            model=model_key, question=q, gold=gold_out, prediction=pred,
            pred_label=pred, correct=bool(corr), image_ref=img_ref, extra=extra))
    out = os.path.join(run_dir, f"{model_key}__{spec.key}.jsonl")
    write_jsonl(out, records)
    stats = {"dataset": spec.key, "model": model_key, "n": n, "correct": correct,
             "accuracy": correct / n if n else 0.0,
             "inferred_correctness": unknown, "predictions": out}
    return out, stats
