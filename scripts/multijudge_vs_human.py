"""Run a panel of LLM judges over the SAME errors a human annotated, then compute,
for each judge: human-vs-judge Cohen's kappa (8-way) + per-family agreement; plus
inter-judge kappa and a majority-vote ensemble vs the human.

Reuses the taxonomy + judge machinery; text-only (deepseek/OpenRouter design),
2048-token budget, multiple-choice gold enriched with option text from choices.json.

    OPENAI_API_KEY=sk-or-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
      python scripts/multijudge_vs_human.py --run main2b \
        --annotations human_annotations-5_updated.json \
        --judges deepseek/deepseek-v4-flash,openai/gpt-4o,qwen/qwen-2.5-72b-instruct,\
meta-llama/llama-3.3-70b-instruct,mistralai/mistral-large,deepseek/deepseek-chat
"""
import argparse, json, os, sys
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench.judge import read_jsonl
from failuremodebench.multijudge import judge_with_model, cohen_kappa, compare


def enrich(rows, choices):
    for e in rows:
        ch = choices.get(e.get("sample_id", ""))
        if ch and ch.get("options"):
            g = str(e.get("gold", "")).strip()
            gt = ch.get("answer_text") or ch["options"].get(g, "")
            if gt and "(" not in str(e.get("gold", "")):
                e["gold"] = f"{e['gold']} ({gt})"
            e["question"] = e.get("question", "") + "\nAnswer options: " + \
                "; ".join(f"{L}) {t}" for L, t in ch["options"].items())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--judges", required=True, help="comma-separated OpenRouter model ids")
    a = ap.parse_args()
    run_dir = os.path.join(HERE, "results", a.run)
    out_dir = os.path.join(run_dir, "multijudge_human")
    os.makedirs(out_dir, exist_ok=True)

    human = {r["error_key"]: r["human"] for r in json.load(open(a.annotations)) if r.get("human")}
    corp = {r["error_key"]: r for r in read_jsonl(os.path.join(run_dir, "corpus.jsonl"))}
    choices = json.load(open(os.path.join(run_dir, "choices.json"))) \
        if os.path.exists(os.path.join(run_dir, "choices.json")) else {}
    sampled = enrich([corp[k] for k in human if k in corp], choices)
    print(f"{len(sampled)} human-annotated errors; judging with {a.judges.count(',')+1} models")

    judges = [m.strip() for m in a.judges.split(",") if m.strip()]
    verdicts = {}  # model -> {error_key: mode}
    for m in judges:
        safe = m.replace("/", "_")
        out = os.path.join(out_dir, f"corpus_{safe}.jsonl")
        print(f"  judging with {m} ...", flush=True)
        try:
            res = judge_with_model(sampled, m, out, provider_name="openai", workers=6)
            verdicts[m] = res
        except Exception as e:
            print(f"    FAILED {m}: {e}")

    # per-judge vs human
    print(f"\n{'judge':40} {'n':>4} {'raw':>6} {'kappa':>6}  {'NA':>3}")
    rows_out = []
    for m, res in verdicts.items():
        pairs = [(human[k], res[k]) for k in human if k in res and res[k] != "NA"]
        na = sum(1 for k in human if res.get(k) == "NA")
        if not pairs:
            print(f"{m:40} {0:>4}  (all NA/failed)"); continue
        h = [x for x, _ in pairs]; j = [y for _, y in pairs]
        raw = sum(x == y for x, y in pairs) / len(pairs)
        kap = cohen_kappa(h, j)
        print(f"{m:40} {len(pairs):>4} {raw*100:5.1f}% {kap:6.3f}  {na:>3}")
        # per-family
        byf = defaultdict(lambda: [0, 0])
        for k in human:
            if k in res and res[k] != "NA":
                byf[corp[k]["family"]][1] += 1
                byf[corp[k]["family"]][0] += int(human[k] == res[k])
        rows_out.append({"judge": m, "n": len(pairs), "raw_agreement": round(raw, 3),
                         "cohen_kappa": round(kap, 3), "na": na,
                         "per_family": {f: round(v[0]/v[1], 3) for f, v in byf.items() if v[1]}})

    # ensemble (majority vote across judges) vs human
    keys = [k for k in human if all(k in v for v in verdicts.values())]
    ens = {}
    for k in keys:
        votes = Counter(verdicts[m][k] for m in verdicts if verdicts[m][k] != "NA")
        if votes:
            ens[k] = votes.most_common(1)[0][0]
    epairs = [(human[k], ens[k]) for k in ens]
    if epairs:
        eh = [x for x, _ in epairs]; ej = [y for _, y in epairs]
        eraw = sum(x == y for x, y in epairs) / len(epairs)
        ekap = cohen_kappa(eh, ej)
        print(f"\n{'ENSEMBLE (majority vote)':40} {len(epairs):>4} {eraw*100:5.1f}% {ekap:6.3f}")

    # inter-judge kappa (reuse compare)
    inter = compare({m: v for m, v in verdicts.items()}, out_dir)
    kv = [d["cohen_kappa"] for d in inter["pairwise_agreement"].values()]
    if kv:
        print(f"\ninter-judge pairwise kappa: mean {sum(kv)/len(kv):.3f} "
              f"(range {min(kv):.3f}-{max(kv):.3f}), {len(kv)} pairs")

    summary = {"n_errors": len(human), "vs_human": rows_out,
               "ensemble_vs_human": {"n": len(epairs), "raw": round(eraw, 3),
                                     "cohen_kappa": round(ekap, 3)} if epairs else None,
               "inter_judge_mean_kappa": round(sum(kv)/len(kv), 3) if kv else None}
    json.dump(summary, open(os.path.join(out_dir, "vs_human_summary.json"), "w"), indent=2)
    print(f"\nwrote {out_dir}/vs_human_summary.json")


if __name__ == "__main__":
    main()
