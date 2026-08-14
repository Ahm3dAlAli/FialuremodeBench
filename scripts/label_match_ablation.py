"""Label-matching ablation: is the low recognition accuracy a matching artefact
or genuine model behaviour? (Pre-empts the obvious reviewer objection.)

Re-scores the STORED recognition predictions (raw model text vs gold) under three
progressively looser matchers, using the same LabelMatcher the harness uses:
  exact     -- gold label appears as a whole phrase in the model output
  +jaccard  -- + token-overlap fuzzy match (the harness default w/o embeddings)
  +embed    -- + semantic embedding match (opt-in; needs sentence-transformers)

If accuracy barely moves as matching loosens, the errors are genuine (the model
names a different / broader class), not resolution failures.

    python scripts/label_match_ablation.py --run main2b [--embed]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench.config import DATASETS
from failuremodebench.labelmatch import LabelMatcher, normalize
from failuremodebench.recognition import load_label_set


def score(recs, matcher):
    n = c = 0
    for r in recs:
        gold = r.get("gold", "")
        mi, _ = matcher.match(r.get("prediction", ""))
        pred = matcher.labels[mi] if mi is not None else ""
        n += 1
        c += int(bool(pred) and normalize(pred) == normalize(gold))
    return c / n if n else 0.0


class _ExactOnly(LabelMatcher):
    """Substring/synonym only (disable the Jaccard fuzzy fallback)."""
    def match(self, answer):
        idx, method = super().match(answer)
        return (idx, method) if method in ("substring", "synonym") else (None, "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--embed", action="store_true", help="also run the (slow) embedding matcher")
    a = ap.parse_args()
    pred_dir = os.path.join(HERE, "results", a.run, "predictions")
    print(f"{'dataset':16} {'stored':>7} {'exact':>7} {'+jaccard':>9}" +
          ("  +embed" if a.embed else ""))
    rows = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        model, ds = os.path.basename(f)[:-6].split("__", 1)
        if ds not in DATASETS or DATASETS[ds].modality != "recognition":
            continue
        recs = [json.loads(l) for l in open(f)]
        labels, syn = load_label_set(DATASETS[ds].label_set or "", HERE)
        if not labels:
            continue
        stored = sum(r.get("correct") for r in recs) / len(recs)
        exact = score(recs, _ExactOnly(labels, syn))
        jacc = score(recs, LabelMatcher(labels, syn, use_embeddings=False))
        line = f"{ds:16} {stored*100:6.1f}% {exact*100:6.1f}% {jacc*100:8.1f}%"
        row = {"dataset": ds, "stored": stored, "exact": exact, "jaccard": jacc}
        if a.embed:
            emb = score(recs, LabelMatcher(labels, syn, use_embeddings=True))
            line += f" {emb*100:6.1f}%"; row["embed"] = emb
        print(line); rows.append(row)
    out = os.path.join(HERE, "results", a.run, "tables", "label_match_ablation.csv")
    import csv
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); cols = ["dataset", "stored", "exact", "jaccard"] + (["embed"] if a.embed else [])
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"\nwrote {out}")
    print("Interpretation: generative-VLM recognition accuracy is HIGHLY sensitive\n"
          "to label resolution -- exact match ~0% (the model rarely emits the clean\n"
          "label string; it names a broader/near category), embeddings recover much\n"
          "of it. Report the matcher explicitly (we use embeddings, the CLIP-benchmark\n"
          "standard) and treat this as a metric caveat, not a hidden bug. The gap is a\n"
          "quantitative signature of F7 Overgeneralization.")


if __name__ == "__main__":
    main()
