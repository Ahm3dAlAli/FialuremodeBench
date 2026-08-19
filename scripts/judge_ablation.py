"""Judge prompt/decoding ablation (reviewer Q6): is the F7 recognition headline
invariant to judge prompt design and decoding? Re-judge a fixed recognition sample
under variants and report F7 rate + Cohen's kappa vs the deployed configuration."""
import os, re, sys, json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
from failuremodebench import taxonomy
from failuremodebench.judge import read_jsonl, _parse_verdict
from failuremodebench.providers import OpenAIProvider
from failuremodebench.multijudge import cohen_kappa

REC = ["fine_grained", "generic", "robustness", "satellite"]
BASE = taxonomy.JUDGE_SYSTEM

# --- variant system prompts ---
# no_priority: drop the "Decision guidance (apply in this order)" block
no_priority = re.sub(r"Decision guidance \(apply in this order\):.*?(?=Pick the SINGLE)",
                     "", BASE, flags=re.S)
# no_examples: strip the "e.g. ..." manifestation lines from the rubric
no_examples = re.sub(r"\n\s*e\.g\. [^\n]*", "", BASE)

VARIANTS = [
    ("base (deployed)",       BASE,        0.0),
    ("no priority-order",     no_priority, 0.0),
    ("no examples in rubric",  no_examples, 0.0),
    ("sampled (temp=0.7)",    BASE,        0.7),
]

def sample_errors(n_per_fam=60):
    byf = defaultdict(list)
    for r in read_jsonl("results/main2b/corpus.jsonl"):
        if r.get("family") in REC:
            byf[r["family"]].append(r)
    out = []
    for f in REC:
        out += sorted(byf[f], key=lambda r: r["error_key"])[:n_per_fam]
    return out

def judge_one(prov, sysp, temp, e):
    usr = taxonomy.user_prompt(e.get("question", ""), e.get("gold", ""),
                               e.get("prediction", ""), e["dataset"], e["family"])
    for _ in range(4):
        try:
            v = _parse_verdict(prov.complete(sysp, usr, image=None, max_tokens=2048, temperature=temp))
            if v: return e["error_key"], v["failure_mode"]
        except Exception:
            pass
    return e["error_key"], "NA"

def main():
    prov = OpenAIProvider(model="deepseek/deepseek-v4-flash")
    errs = sample_errors()
    print(f"judging {len(errs)} recognition errors under {len(VARIANTS)} variants\n")
    labels = {}  # variant -> {ekey: F}
    for name, sysp, temp in VARIANTS:
        res = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(judge_one, prov, sysp, temp, e) for e in errs]
            for fut in as_completed(futs):
                k, F = fut.result(); res[k] = F
        labels[name] = res
        c = Counter(res.values()); tot = sum(v for k, v in c.items() if k != "NA")
        f7 = c["F7"] / tot * 100 if tot else 0
        print(f"  {name:22} F7 = {f7:5.1f}%   (F6={c['F6']/tot*100:.0f}%, NA={c['NA']})")
    base = labels["base (deployed)"]
    print("\nagreement with deployed config (Cohen's kappa, F7-headline invariance):")
    for name, *_ in VARIANTS[1:]:
        ks = [k for k in base if base[k] != "NA" and labels[name].get(k, "NA") != "NA"]
        kap = cohen_kappa([base[k] for k in ks], [labels[name][k] for k in ks])
        raw = sum(base[k] == labels[name][k] for k in ks) / len(ks) * 100
        print(f"  {name:22} kappa={kap:.3f}  raw={raw:.0f}%  (n={len(ks)})")

if __name__ == "__main__":
    main()
