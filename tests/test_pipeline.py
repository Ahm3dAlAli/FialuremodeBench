"""Offline end-to-end validation of the FailureModeBench post-inference pipeline.

Exercises error-gathering -> stratified sampling -> (stub) judge -> aggregation
-> figures WITHOUT a GPU, network or API key. Run directly:

    python tests/test_pipeline.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from failuremodebench import aggregate, figures, taxonomy
from failuremodebench.config import DATASETS
from failuremodebench.judge import gather_errors, judge_errors, sample_per_family
from failuremodebench.providers import EchoProvider
from failuremodebench.records import PredictionRecord, read_jsonl, write_jsonl

RUN = os.path.join(ROOT, "results", "_selftest")
PRED = os.path.join(RUN, "predictions")


def build_fixture():
    os.makedirs(PRED, exist_ok=True)
    models = ["qwen2vl_7b", "llava16_7b"]
    dsets = ["imagenet", "food101", "dtd", "ai2d", "mathvista", "crpe"]
    for m in models:
        for d in dsets:
            spec = DATASETS[d]
            recs = []
            for i in range(30):
                wrong = (i % 5 < 2)  # 40% error rate
                recs.append(PredictionRecord(
                    sample_id=f"{d}-{i:04d}", dataset=d, family=spec.family,
                    modality=spec.modality, model=m,
                    question=f"What is shown? ({d} sample {i})",
                    gold="sparrow" if d == "imagenet" else f"gold_{i%7}",
                    prediction=("bird" if d == "imagenet" else f"pred_{i%9}")
                                if wrong else ("sparrow" if d == "imagenet" else f"gold_{i%7}"),
                    pred_label=("bird" if wrong else "sparrow") if d == "imagenet" else "",
                    correct=not wrong, image_ref=""))
            write_jsonl(os.path.join(PRED, f"{m}__{d}.jsonl"), recs)


def varied_corpus(sampled):
    """Replace the echo corpus with all-8-modes variety so aggregation/figures
    are meaningfully exercised (echo alone would be 100% F8)."""
    import hashlib
    path = os.path.join(RUN, "corpus.jsonl")
    rows = []
    # family -> a plausible dominant mode (mirrors the paper's claims)
    fam_bias = {"generic": "F7", "fine_grained": "F7", "robustness": "F3",
                "chart": "F4", "math": "F5", "compositional": "F8"}
    for k, e in enumerate(sampled):
        fam = e["family"]
        fm = fam_bias.get(fam, taxonomy.CODES[k % 8])
        if k % 4 == 0:  # inject spread
            fm = taxonomy.CODES[k % 8]
        ekey = hashlib.sha1(f"{e['model']}|{e['dataset']}|{e['sample_id']}".encode()).hexdigest()[:16]
        rows.append({"error_key": ekey, "model": e["model"], "dataset": e["dataset"],
                     "family": fam, "modality": e["modality"], "sample_id": e["sample_id"],
                     "question": e["question"], "gold": e["gold"], "prediction": e["prediction"],
                     "had_image": False, "failure_mode": fm, "confidence": 0.8,
                     "rationale": "synthetic", "secondary_mode": "none"})
    write_jsonl(path, rows)
    return path


def main():
    build_fixture()

    # 1. gather + sample
    errors = gather_errors(PRED)
    assert errors and all(not e["correct"] for e in errors), "gather_errors broken"
    sampled = sample_per_family(errors, n_per_family=50)
    fams = {e["family"] for e in sampled}
    assert fams, "no families sampled"
    print(f"[ok] gathered {len(errors)} errors, sampled {len(sampled)} across {len(fams)} families")

    # 2. echo judge (plumbing: parse + incremental write + resume)
    corpus = os.path.join(RUN, "corpus.jsonl")
    if os.path.exists(corpus):
        os.remove(corpus)
    counts = judge_errors(sampled, EchoProvider(), corpus, attach_images=False)
    assert sum(counts.values()) == len(sampled), (counts, len(sampled))
    assert set(read_jsonl(corpus)[0]).issuperset({"failure_mode", "rationale", "confidence"})
    print(f"[ok] echo judge wrote {sum(counts.values())} verdicts, all parsed")
    # resume: second call adds nothing
    counts2 = judge_errors(sampled, EchoProvider(), corpus, attach_images=False)
    assert len(read_jsonl(corpus)) == len(sampled), "resume double-wrote"
    print("[ok] judge resume is idempotent")

    # 3. aggregate + figures on a varied corpus
    varied_corpus(sampled)
    out_dir = os.path.join(RUN, "tables")
    fig_dir = os.path.join(RUN, "figures")
    summary = aggregate.run_all(PRED, corpus, out_dir)
    for f in ["accuracy.csv", "failure_by_family.csv", "failure_by_dataset.csv",
              "failure_by_model.csv"]:
        assert os.path.exists(os.path.join(out_dir, f)), f
    # percentages per family sum to ~100
    import csv
    with open(os.path.join(out_dir, "failure_by_family.csv")) as fh:
        r = list(csv.reader(fh))
    codes = taxonomy.CODES
    pct_start = 2 + len(codes)
    for row in r[1:]:
        s = sum(float(x) for x in row[pct_start:])
        assert 99.0 <= s <= 101.0, (row[0], s)
    print(f"[ok] aggregation tables written; dominant/family = {summary['dominant_by_family']}")
    assert summary["confusion_files"], "no confusion matrices for fine-grained sets"
    print(f"[ok] confusion matrices: {[os.path.basename(x) for x in summary['confusion_files']]}")

    figs = figures.make_all(out_dir, fig_dir)
    assert all(os.path.exists(x) for x in figs) and figs, figs
    print(f"[ok] figures: {[os.path.basename(x) for x in figs]}")

    print("\nALL PIPELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
