"""Expand the judged corpus to N per family for a larger human study.

Gathers errors from results/<run>/predictions/, takes a deterministic N-per-family
sample, and (in --judge mode) judges the errors not already in corpus.jsonl with
the fixed pipeline (deepseek text-only, 2048 tokens, multiple-choice gold enriched
from choices.json). Reuses existing verdicts (skips by error_key), so re-running is
cheap. Two stages so images/choices can be fetched from rolf in between:

  1) python scripts/expand_corpus.py --run main2b --n 100 --emit-needed
     -> writes results/<run>/images_needed.txt for the whole sample
     (then: scripts/pull_study_images.sh + rolf extract for choices.json)

  2) OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
     python scripts/expand_corpus.py --run main2b --n 100 --judge
     -> judges the new errors, appends to corpus.jsonl
"""
import argparse, glob, hashlib, json, os, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench import taxonomy
from failuremodebench.config import DATASETS
from failuremodebench.judge import read_jsonl, sample_per_family, _parse_verdict
from failuremodebench.providers import OpenAIProvider


def ek(e):
    return hashlib.sha1(f"{e['model']}|{e['dataset']}|{e['sample_id']}".encode()).hexdigest()[:16]


def gather(run_dir):
    errs = []
    for f in glob.glob(os.path.join(run_dir, "predictions", "*__*.jsonl")):
        ds = os.path.basename(f).split("__", 1)[1][:-6]
        if ds not in DATASETS:
            continue
        for r in read_jsonl(f):
            if not r.get("correct"):
                r.setdefault("family", DATASETS[ds].family)
                r.setdefault("modality", DATASETS[ds].modality)
                errs.append(r)
    return errs


def judge_one(prov, e, choices):
    q = e.get("question", "")
    gold = str(e.get("gold", ""))
    ch = choices.get(e.get("sample_id", ""))
    if ch and ch.get("options"):
        gt = ch.get("answer_text") or ch["options"].get(gold.strip(), "")
        if gt and "(" not in gold:
            gold = f"{gold} ({gt})"
        q = q + "\nAnswer options: " + "; ".join(f"{L}) {t}" for L, t in ch["options"].items())
    usr = taxonomy.user_prompt(q, gold, e.get("prediction", ""), e["dataset"], e["family"])
    for attempt in range(4):
        try:
            v = _parse_verdict(prov.complete(taxonomy.JUDGE_SYSTEM, usr, image=None,
                                             max_tokens=2048, temperature=0.0))
            if v:
                return v
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {"failure_mode": "NA", "confidence": 0.0, "rationale": "unparseable", "secondary_mode": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--emit-needed", action="store_true")
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    run_dir = os.path.join(HERE, "results", a.run)
    corpus_path = os.path.join(run_dir, "corpus.jsonl")
    errors = gather(run_dir)
    sample = sample_per_family(errors, a.n)
    print(f"sample: {len(sample)} errors @ {a.n}/family "
          f"({dict((f, sum(1 for e in sample if e['family']==f)) for f in sorted({e['family'] for e in sample}))})")

    if a.emit_needed:
        names = sorted({f"{e['model']}-{e['sample_id']}.png" for e in sample})
        open(os.path.join(run_dir, "images_needed.txt"), "w").write("\n".join(names) + "\n")
        print(f"wrote images_needed.txt: {len(names)} images for the full sample")

    if a.judge:
        done = {r["error_key"] for r in read_jsonl(corpus_path)}
        todo = [e for e in sample if ek(e) not in done]
        print(f"{len(sample)-len(todo)} already judged; judging {len(todo)} new errors with {a.model}")
        cpath = os.path.join(run_dir, "choices.json")
        choices = json.load(open(cpath)) if os.path.exists(cpath) else {}
        prov = OpenAIProvider(model=a.model)
        from collections import Counter
        cnt = Counter()
        with open(corpus_path, "a", encoding="utf-8") as fout, \
                ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(judge_one, prov, e, choices): e for e in todo}
            n = 0
            for fut in as_completed(futs):
                e = futs[fut]; v = fut.result()
                row = {"error_key": ek(e), "model": e["model"], "dataset": e["dataset"],
                       "family": e["family"], "modality": e.get("modality", "vqa"),
                       "sample_id": e["sample_id"], "question": e.get("question", ""),
                       "gold": e.get("gold", ""), "prediction": e.get("prediction", ""),
                       "had_image": False, **v}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
                cnt[v["failure_mode"]] += 1; n += 1
                if n % 50 == 0:
                    print(f"  judged {n}/{len(todo)} ...", flush=True)
        print(f"done: {n} judged, verdicts {dict(cnt)}")
        tot = read_jsonl(corpus_path)
        print(f"corpus now {len(tot)} errors, NA={sum(1 for r in tot if str(r.get('failure_mode'))=='NA')}")


if __name__ == "__main__":
    main()
