"""Multi-annotator judge-validity analysis (reproducible).

Reads every annotator file in human_annotations/*.json, joins to the judged corpus,
and reports: per-annotator agreement with the LLM judge, human-human Cohen's kappa,
Fleiss' kappa (humans, and humans+judge), the judge's agreement with human consensus
(per family), and the pairwise kappa matrix. This is the paper's §Judge-Validity.

    python scripts/multi_annotator_agreement.py --run main2b
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench.judge import read_jsonl
from failuremodebench.multijudge import cohen_kappa

CODES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]


def fleiss(rater_dicts, keys):
    """Fleiss' kappa over the given keys for a list of key->label dicts."""
    N, n = len(keys), len(rater_dicts)
    P = []
    for k in keys:
        cnt = Counter(r[k] for r in rater_dicts)
        P.append((sum(v * v for v in cnt.values()) - n) / (n * (n - 1)))
    Pbar = sum(P) / N
    cat = Counter(r[k] for k in keys for r in rater_dicts)
    tot = N * n
    Pe = sum((v / tot) ** 2 for v in cat.values())
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--annotations-dir", default=os.path.join(HERE, "human_annotations"))
    a = ap.parse_args()
    corp = {r["error_key"]: r for r in read_jsonl(os.path.join(HERE, "results", a.run, "corpus.jsonl"))}

    # load annotators (exclude GW mislabel flags from the F1-F8 agreement)
    raters = {}
    gw = {}
    for f in sorted(glob.glob(os.path.join(a.annotations_dir, "*.json"))):
        name = os.path.basename(f)[:-5]
        rows = json.load(open(f))
        raters[name] = {r["error_key"]: r["human"] for r in rows
                        if r.get("human") and r["human"] != "GW"}
        gw[name] = {r["error_key"] for r in rows if r.get("human") == "GW"}
    judge = {k: v["failure_mode"] for k, v in corp.items() if v.get("failure_mode") in CODES}
    print(f"annotators: {list(raters)}")

    # per-annotator vs judge
    print("\n== each annotator vs LLM judge ==")
    for name, d in raters.items():
        ks = [k for k in d if k in judge]
        h = [d[k] for k in ks]; j = [judge[k] for k in ks]
        raw = sum(x == y for x, y in zip(h, j)) / len(ks)
        print(f"  {name:20} n={len(ks):4}  raw={raw*100:.1f}%  kappa={cohen_kappa(h,j):.3f}"
              f"  (GW flags: {len(gw[name])})")

    # pairwise kappa matrix (humans + judge)
    allr = dict(raters); allr["JUDGE"] = judge
    labs = list(allr)
    print("\n== pairwise Cohen's kappa ==")
    print(" " * 22 + "".join(f"{l[:10]:>11}" for l in labs))
    for x in labs:
        row = f"  {x[:20]:20}"
        for y in labs:
            ks = [k for k in allr[x] if k in allr[y]]
            row += f"{cohen_kappa([allr[x][k] for k in ks],[allr[y][k] for k in ks]):>11.3f}" if ks else f"{'-':>11}"
        print(row)

    # items rated by all humans (+judge), Fleiss + consensus
    hum = list(raters)
    keys = [k for k in raters[hum[0]] if all(k in raters[h] for h in hum) and k in judge]
    if len(hum) >= 2 and keys:
        fh = fleiss([{k: raters[h][k] for k in keys} for h in hum], keys)
        fhj = fleiss([{k: raters[h][k] for k in keys} for h in hum] + [{k: judge[k] for k in keys}], keys)
        print(f"\n== Fleiss' kappa (n={len(keys)} shared items) ==")
        print(f"  humans only:        {fh:.3f}")
        print(f"  humans + LLM judge: {fhj:.3f}")

        cons = [k for k in keys if len({raters[h][k] for h in hum}) == 1]
        match = sum(1 for k in cons if judge[k] == raters[hum[0]][k])
        print(f"\n== judge vs human consensus (both/all humans agree, n={len(cons)}) ==")
        print(f"  overall: {match}/{len(cons)} = {match/len(cons)*100:.1f}%")
        byf = defaultdict(lambda: [0, 0])
        for k in cons:
            fam = corp[k].get("family", "?"); byf[fam][1] += 1
            byf[fam][0] += int(judge[k] == raters[hum[0]][k])
        for fam in sorted(byf):
            c, n = byf[fam]
            print(f"    {fam:14} {c}/{n} = {c/n*100:.0f}%")


if __name__ == "__main__":
    main()
