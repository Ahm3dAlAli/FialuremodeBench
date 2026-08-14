"""Turn the annotated error corpus + prediction files into the paper's tables.

Produces:
  * accuracy.csv           -- top-1 accuracy / VQA score per (model, dataset)
  * main_results.csv       -- Table 3: per-model Recog.Acc / VQA Score / Macro-F1
  * macro_f1.csv           -- macro-F1 per (model, fine-grained dataset)
  * failure_by_family.csv  -- F1..F8 distribution (%) per task family  [Table 4/headline]
  * failure_by_dataset.csv -- F1..F8 distribution per dataset
  * failure_by_model.csv   -- F1..F8 distribution per model
  * confusion/<ds>.csv     -- per-class confusion for fine-grained recognition
  * counts_check.csv       -- asserts evaluated counts vs Table 2 (drift guard)
All are plain CSVs so they drop straight into LaTeX / pandas.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict

from . import taxonomy
from .config import DATASETS, FAMILY_DISPLAY, TASK_FAMILIES
from .records import read_jsonl


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _iter_pred_files(run_dir):
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith(".jsonl") and "__" in fn and fn != "corpus.jsonl":
            model, dataset = fn[:-6].split("__", 1)
            if dataset in DATASETS:
                yield model, dataset, os.path.join(run_dir, fn)


# --------------------------------------------------------------------------- #
# Accuracy + Macro-F1 (Table 3)                                               #
# --------------------------------------------------------------------------- #
def accuracy_table(run_dir: str, out_dir: str) -> list[dict]:
    per = defaultdict(lambda: [0, 0])
    for model, dataset, path in _iter_pred_files(run_dir):
        for r in read_jsonl(path):
            per[(model, dataset)][1] += 1
            per[(model, dataset)][0] += int(r.get("correct", False))
    rows, table = [], []
    for (model, dataset), (c, n) in sorted(per.items()):
        acc = c / n if n else 0.0
        table.append({"model": model, "dataset": dataset, "n": n,
                      "correct": c, "accuracy": round(acc, 4)})
        rows.append([model, dataset, n, c, round(acc, 4)])
    _write_csv(os.path.join(out_dir, "accuracy.csv"),
               ["model", "dataset", "n", "correct", "accuracy"], rows)
    return table


def _macro_f1(pairs) -> float:
    """macro-F1 over classes present as gold, from (gold, pred) pairs."""
    tp = Counter(); fp = Counter(); fn = Counter()
    classes = set()
    for g, p in pairs:
        classes.add(g)
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1
    f1s = []
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def macro_f1_table(run_dir: str, out_dir: str) -> dict:
    """macro-F1 per (model, fine-grained dataset)."""
    fine = {k for k, d in DATASETS.items() if d.fine_grained}
    per = {}
    for model, dataset, path in _iter_pred_files(run_dir):
        if dataset not in fine:
            continue
        pairs = [(r.get("gold", "?"), r.get("pred_label", "") or "<none>")
                 for r in read_jsonl(path)]
        per[(model, dataset)] = round(_macro_f1(pairs), 4)
    _write_csv(os.path.join(out_dir, "macro_f1.csv"),
               ["model", "dataset", "macro_f1"],
               [[m, d, v] for (m, d), v in sorted(per.items())])
    return per


def main_results_table(run_dir: str, out_dir: str) -> list:
    """Table 3: per model, mean recog accuracy, mean VQA score, mean macro-F1."""
    acc = accuracy_table(run_dir, out_dir)
    mf1 = macro_f1_table(run_dir, out_dir)
    recog = defaultdict(list); vqa = defaultdict(list); f1 = defaultdict(list)
    for r in acc:
        mod = DATASETS[r["dataset"]].modality
        (recog if mod == "recognition" else vqa)[r["model"]].append(r["accuracy"])
    for (m, _), v in mf1.items():
        f1[m].append(v)
    models = sorted(set(list(recog) + list(vqa) + list(f1)))
    rows = []
    for m in models:
        ra = round(sum(recog[m]) / len(recog[m]), 4) if recog[m] else None
        vs = round(sum(vqa[m]) / len(vqa[m]), 4) if vqa[m] else None
        mf = round(sum(f1[m]) / len(f1[m]), 4) if f1[m] else None
        rows.append([m, ra, vs, mf])
    _write_csv(os.path.join(out_dir, "main_results.csv"),
               ["model", "recog_acc", "vqa_score", "macro_f1"], rows)
    return rows


# --------------------------------------------------------------------------- #
# Failure-mode distributions (Table 4)                                        #
# --------------------------------------------------------------------------- #
def _dist_rows(corpus: list[dict], key: str, order=None, display=None):
    groups: dict[str, Counter] = defaultdict(Counter)
    for r in corpus:
        fm = r.get("failure_mode", "NA")
        if fm in taxonomy.CODES:
            groups[r.get(key, "?")][fm] += 1
    header = [key, "n_errors"] + taxonomy.CODES + [f"{c}_pct" for c in taxonomy.CODES]
    keys = [g for g in (order or []) if g in groups] + \
           sorted(g for g in groups if not order or g not in order)
    rows = []
    for g in keys:
        cnt = groups[g]
        total = sum(cnt.values())
        counts = [cnt.get(c, 0) for c in taxonomy.CODES]
        pcts = [round(100 * x / total, 1) if total else 0.0 for x in counts]
        rows.append([(display or {}).get(g, g), total] + counts + pcts)
    return header, rows


def failure_tables(corpus_path: str, out_dir: str) -> dict:
    corpus = read_jsonl(corpus_path)
    result = {}
    specs = [("family", "failure_by_family.csv", TASK_FAMILIES, FAMILY_DISPLAY),
             ("dataset", "failure_by_dataset.csv", list(DATASETS), None),
             ("model", "failure_by_model.csv", None, None)]
    for key, fname, order, display in specs:
        header, rows = _dist_rows(corpus, key, order, display)
        _write_csv(os.path.join(out_dir, fname), header, rows)
        result[key] = {"header": header, "rows": rows}
    dom = {}
    for row in result["family"]["rows"]:
        pct_slice = row[2 + len(taxonomy.CODES):]
        if pct_slice and row[1]:
            top = max(range(len(taxonomy.CODES)), key=lambda i: pct_slice[i])
            dom[row[0]] = (taxonomy.CODES[top], pct_slice[top])
    result["dominant_by_family"] = dom
    return result


def confusion_matrices(run_dir: str, out_dir: str) -> list[str]:
    written = []
    fine = {k for k, d in DATASETS.items() if d.fine_grained}
    for model, dataset, path in _iter_pred_files(run_dir):
        if dataset not in fine:
            continue
        pairs, labels = Counter(), set()
        for r in read_jsonl(path):
            g, p = r.get("gold", "?"), r.get("pred_label", "") or "<none>"
            pairs[(g, p)] += 1
            labels.update([g, p])
        labs = sorted(labels)
        idx = {l: i for i, l in enumerate(labs)}
        rows = [[l] + [0] * len(labs) for l in labs]
        for (g, p), c in pairs.items():
            rows[idx[g]][idx[p] + 1] = c
        out = os.path.join(out_dir, "confusion", f"{model}__{dataset}.csv")
        _write_csv(out, ["gold\\pred"] + labs, rows)
        written.append(out)
    return written


# --------------------------------------------------------------------------- #
# Count assertion (paper: "Table 2 cannot silently drift from the code")      #
# --------------------------------------------------------------------------- #
def counts_check(run_dir: str, out_dir: str) -> dict:
    """Compare max evaluated n per dataset against the Table 2 'Used' count."""
    seen = defaultdict(int)
    for _, dataset, path in _iter_pred_files(run_dir):
        seen[dataset] = max(seen[dataset], sum(1 for _ in read_jsonl(path)))
    rows, mismatches = [], []
    for k, d in DATASETS.items():
        got = seen.get(k, 0)
        ok = (got == d.used_size)
        rows.append([k, got, d.used_size, d.original_size, "OK" if ok else
                     ("PARTIAL" if 0 < got < d.used_size else "MISSING")])
        if got and not ok:
            mismatches.append((k, got, d.used_size))
    _write_csv(os.path.join(out_dir, "counts_check.csv"),
               ["dataset", "evaluated_n", "used_expected", "orig", "status"], rows)
    return {"mismatches": mismatches, "checked": len(rows)}


def run_all(run_dir: str, corpus_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    main = main_results_table(run_dir, out_dir)   # also writes accuracy + macro_f1
    fails = failure_tables(corpus_path, out_dir) if os.path.exists(corpus_path) else {}
    conf = confusion_matrices(run_dir, out_dir)
    counts = counts_check(run_dir, out_dir)
    return {"models": len(main), "confusion_files": conf,
            "dominant_by_family": fails.get("dominant_by_family", {}),
            "count_mismatches": counts["mismatches"]}
