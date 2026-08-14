"""Human vs LLM-judge agreement on the 8-way F1-F8 taxonomy (ICLR validation).

Consumes the human_annotations.json exported from the study page and computes
raw agreement, Cohen's kappa (8-way), per-family agreement, and the human/judge
confusion so you can report exactly where the judge diverges from a human.

    python scripts/human_agreement.py --run main2b --annotations human_annotations.json
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench import taxonomy
from failuremodebench.multijudge import cohen_kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--annotations", required=True)
    a = ap.parse_args()
    ann = json.load(open(a.annotations))
    # "GW" = annotator judged the dataset gold wrong / model actually correct; it is
    # label noise, not an F1-F8 model failure, so exclude it from kappa and report it.
    labeled = [r for r in ann if r.get("human")]
    gw = [r for r in labeled if r["human"] == "GW"]
    if labeled:
        print(f"mislabel flags (gold wrong / model correct): {len(gw)}/{len(labeled)} "
              f"= {len(gw)/len(labeled)*100:.1f}% of labeled errors")
        if gw:
            byd = Counter(r.get("dataset", r.get("family", "?")) for r in gw)
            print(f"  by dataset/family: {dict(byd)}\n")
    pairs = [(r["human"], r["judge"], r.get("family", "?"))
             for r in ann if r.get("human") and r.get("human") != "GW"
             and r.get("judge") and str(r.get("judge")) != "NA"]
    if not pairs:
        print("no labeled+judged pairs found"); return
    h = [p[0] for p in pairs]; j = [p[1] for p in pairs]

    raw = sum(x == y for x, y in zip(h, j)) / len(pairs)
    kappa = cohen_kappa(h, j)
    print(f"n = {len(pairs)} labeled errors")
    print(f"raw agreement (human vs deepseek judge): {raw*100:.1f}%")
    print(f"Cohen's kappa (8-way F1-F8):             {kappa:.3f}")

    print("\nper-family agreement:")
    byf = defaultdict(lambda: [0, 0])
    for hh, jj, fam in pairs:
        byf[fam][1] += 1; byf[fam][0] += int(hh == jj)
    for fam in sorted(byf):
        c, n = byf[fam]
        print(f"  {fam:14} {c}/{n} = {c/n*100:.0f}%")

    print("\nhuman -> judge confusion (rows=human, cols=judge):")
    conf = Counter((hh, jj) for hh, jj in zip(h, j))
    codes = taxonomy.CODES
    print("      " + " ".join(c.rjust(4) for c in codes))
    for hc in codes:
        row = " ".join(str(conf.get((hc, jc), 0)).rjust(4) for jc in codes)
        if any(conf.get((hc, jc), 0) for jc in codes):
            print(f"  {hc}  {row}")

    out = os.path.join(HERE, "results", a.run, "human_agreement.json")
    json.dump({"n": len(pairs), "raw_agreement": round(raw, 4),
               "cohen_kappa": round(kappa, 4),
               "per_family": {f: round(byf[f][0]/byf[f][1], 3) for f in byf}},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
