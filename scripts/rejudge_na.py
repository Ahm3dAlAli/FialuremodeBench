"""Re-judge the corpus errors the judge left as 'NA' (unparseable reply).

Root cause: multiple-choice VQA errors were sent to the judge with a BARE LETTER
gold ("gold: C") and no option text, so the judge could not classify them and
emitted unparseable output -> NA. This re-judges only the NA rows, injecting the
option text from choices.json (same enrichment the human study got), and patches
results/<run>/corpus.jsonl in place (a .bak is written first).

    OPENAI_API_KEY=sk-or-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
      python scripts/rejudge_na.py --run main2b --model deepseek/deepseek-v4-flash
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench import taxonomy
from failuremodebench.judge import _parse_verdict, _load_image, read_jsonl
from failuremodebench.providers import OpenAIProvider


def enrich_gold(e, choices):
    ch = choices.get(e.get("sample_id", ""))
    gold = str(e.get("gold", ""))
    if not ch or not ch.get("options"):
        return gold, ""
    opts = ch["options"]
    gtext = ch.get("answer_text") or opts.get(gold.strip(), "")
    olist = "; ".join(f"{L}) {t}" for L, t in opts.items())
    return (f"{gold} ({gtext})" if gtext else gold), olist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--img-dir", default=None, help="dir of extracted images (default results/<run>/images)")
    a = ap.parse_args()
    run_dir = os.path.join(HERE, "results", a.run)
    corpus_path = os.path.join(run_dir, "corpus.jsonl")
    img_dir = a.img_dir or os.path.join(run_dir, "images")
    choices = json.load(open(os.path.join(run_dir, "choices.json"))) \
        if os.path.exists(os.path.join(run_dir, "choices.json")) else {}

    rows = read_jsonl(corpus_path)
    na_idx = [i for i, r in enumerate(rows) if str(r.get("failure_mode")) == "NA"]
    print(f"{len(rows)} corpus errors; {len(na_idx)} are NA -> re-judging with "
          f"{a.model} (+choice text)")
    if not na_idx:
        return
    prov = OpenAIProvider(model=a.model)

    fixed = still_na = 0
    from collections import Counter
    newc = Counter()
    for i in na_idx:
        e = rows[i]
        gold, olist = enrich_gold(e, choices)
        q = e.get("question", "")
        if olist:
            q = f"{q}\nAnswer options: {olist}"
        usr = taxonomy.user_prompt(q, gold, e.get("prediction", ""),
                                   e["dataset"], e["family"])
        # The deepseek judge is text-only (OpenRouter rejects image input); the
        # whole corpus was judged text-only, so stay consistent. The real fix for
        # the NA rows is the injected option text above, not an image.
        image = None
        verdict = None
        for attempt in range(4):
            try:
                reply = prov.complete(taxonomy.JUDGE_SYSTEM, usr, image=image,
                                      max_tokens=2048, temperature=0.0)
                verdict = _parse_verdict(reply)
                if verdict:
                    break
            except Exception as ex:
                if attempt == 3:
                    print(f"  {e['dataset']}/{e['sample_id']}: {ex}")
        if verdict:
            rows[i].update(verdict)
            rows[i]["rationale"] = verdict.get("rationale", "")[:400]
            fixed += 1
            newc[verdict["failure_mode"]] += 1
        else:
            still_na += 1
    # write back (with backup)
    os.replace(corpus_path, corpus_path + ".bak")
    with open(corpus_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nre-judged: {fixed} fixed, {still_na} still NA")
    print(f"new labels: {dict(newc)}")
    print(f"patched {corpus_path} (backup: corpus.jsonl.bak)")


if __name__ == "__main__":
    main()
