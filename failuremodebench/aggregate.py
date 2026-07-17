"""Turn the annotated error corpus + prediction files into the paper's tables.

Produces:
  * accuracy.csv          -- top-1 accuracy / VQA score per (model, dataset)
  * failure_by_family.csv -- F1..F8 distribution (%) per task family  [headline]
  * failure_by_dataset.csv-- F1..F8 distribution per dataset
  * failure_by_model.csv  -- F1..F8 distribution per model
  * confusion/<ds>.csv    -- per-class confusion for fine-grained recognition
All are plain CSVs so they drop straight into LaTeX / pandas.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict

from . import taxonomy
from .config import DATASETS, TASK_FAMILIES
from .records import read_jsonl


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def accuracy_table(run_dir: str, out_dir: str) -> list[dict]:
    rows, per = [], defaultdict(lambda: [0, 0])
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith(".jsonl") and "__" in fn:
            model, dataset = fn[:-6].split("__", 1)
            for r in read_jsonl(os.path.join(run_dir, fn)):
                per[(model, dataset)][1] += 1
                per[(model, dataset)][0] += int(r.get("correct", False))
    table = []
    for (model, dataset), (c, n) in sorted(per.items()):
        acc = c / n if n else 0.0
        table.append({"model": model, "dataset": dataset, "n": n,
                      "correct": c, "accuracy": round(acc, 4)})
        rows.append([model, dataset, n, c, round(acc, 4)])
    _write_csv(os.path.join(out_dir, "accuracy.csv"),
               ["model", "dataset", "n", "correct", "accuracy"], rows)
    return table


def _dist_rows(corpus: list[dict], key: str) -> tuple[list, list]:
    """failure-mode distribution grouped by corpus[key]."""
    groups: dict[str, Counter] = defaultdict(Counter)
    for r in corpus:
        fm = r.get("failure_mode", "NA")
        if fm in taxonomy.CODES:
            groups[r.get(key, "?")][fm] += 1
    header = [key, "n_errors"] + taxonomy.CODES + [f"{c}_pct" for c in taxonomy.CODES]
    rows = []
    for g in sorted(groups):
        cnt = groups[g]
        total = sum(cnt.values())
        counts = [cnt.get(c, 0) for c in taxonomy.CODES]
        pcts = [round(100 * x / total, 1) if total else 0.0 for x in counts]
        rows.append([g, total] + counts + pcts)
    return header, rows


def failure_tables(corpus_path: str, out_dir: str) -> dict:
    corpus = read_jsonl(corpus_path)
    result = {}
    for key, fname in [("family", "failure_by_family.csv"),
                       ("dataset", "failure_by_dataset.csv"),
                       ("model", "failure_by_model.csv")]:
        header, rows = _dist_rows(corpus, key)
        _write_csv(os.path.join(out_dir, fname), header, rows)
        result[key] = {"header": header, "rows": rows}
    # dominant mode per family (paper's headline claim)
    dom = {}
    fam_h, fam_rows = result["family"]["header"], result["family"]["rows"]
    for row in fam_rows:
        fam = row[0]
        pct_slice = row[2 + len(taxonomy.CODES):]
        if pct_slice:
            top = max(range(len(taxonomy.CODES)), key=lambda i: pct_slice[i])
            dom[fam] = (taxonomy.CODES[top], pct_slice[top])
    result["dominant_by_family"] = dom
    return result


def confusion_matrices(run_dir: str, out_dir: str) -> list[str]:
    written = []
    fine = {k for k, d in DATASETS.items() if d.fine_grained}
    for fn in sorted(os.listdir(run_dir)):
        if not (fn.endswith(".jsonl") and "__" in fn):
            continue
        model, dataset = fn[:-6].split("__", 1)
        if dataset not in fine:
            continue
        pairs = Counter()
        labels = set()
        for r in read_jsonl(os.path.join(run_dir, fn)):
            g, p = r.get("gold", "?"), r.get("pred_label", "") or "<none>"
            pairs[(g, p)] += 1
            labels.update([g, p])
        labs = sorted(labels)
        idx = {l: i for i, l in enumerate(labs)}
        rows = [[l] + [0] * len(labs) for l in labs]
        for (g, p), c in pairs.items():
            rows[idx[g]][idx[p] + 1] = c
        path = os.path.join(out_dir, "confusion", f"{model}__{dataset}.csv")
        _write_csv(path, ["gold\\pred"] + labs, rows)
        written.append(path)
    return written


def run_all(run_dir: str, corpus_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    acc = accuracy_table(run_dir, out_dir)
    fails = failure_tables(corpus_path, out_dir) if os.path.exists(corpus_path) else {}
    conf = confusion_matrices(run_dir, out_dir)
    return {"accuracy_rows": len(acc), "confusion_files": conf,
            "dominant_by_family": fails.get("dominant_by_family", {})}
