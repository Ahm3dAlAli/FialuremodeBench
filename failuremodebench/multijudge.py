"""Multi-judge study: run several LLM judges over the SAME sampled errors and
compare their F1-F8 verdicts.

Motivation: the failure taxonomy is only as trustworthy as the judge. Running an
independent panel of judges and measuring their agreement (per-family dominant
mode + pairwise Cohen's kappa + a majority-vote ensemble) directly addresses the
judge-reliability question. All judges are OpenAI-compatible (OpenRouter here),
so this runs locally with no GPU. API calls are issued concurrently for speed.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import taxonomy
from .judge import _parse_verdict, gather_errors, sample_per_family
from .providers import get_provider


def _judge_one(provider, e: dict) -> str:
    sys = taxonomy.JUDGE_SYSTEM
    usr = taxonomy.user_prompt(e.get("question", ""), e.get("gold", ""),
                               e.get("prediction", ""), e["dataset"], e["family"])
    for _ in range(3):
        try:
            # 2048 (not 400): reasoning judges spend the budget on a hidden
            # reasoning pass and emit empty content under a small cap -> NA.
            v = _parse_verdict(provider.complete(sys, usr, image=None,
                                                 max_tokens=2048, temperature=0.0))
            if v:
                return v["failure_mode"]
        except Exception:
            pass
    return "NA"


def judge_with_model(sampled: list[dict], model: str, out_path: str,
                     provider_name="openai", workers=8) -> dict:
    """Judge all sampled errors with one model (concurrent). Returns {error_key: mode}."""
    done = {}
    if os.path.exists(out_path):
        for r in (json.loads(l) for l in open(out_path)):
            done[r["error_key"]] = r["failure_mode"]
    provider = get_provider(provider_name, model)
    todo = [e for e in sampled if _ekey(e) not in done]
    results = dict(done)
    with open(out_path, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_judge_one, provider, e): e for e in todo}
        for fut in as_completed(futs):
            e = futs[fut]; mode = fut.result()
            k = _ekey(e)
            results[k] = mode
            f.write(json.dumps({"error_key": k, "family": e["family"],
                                "dataset": e["dataset"], "failure_mode": mode}) + "\n")
            f.flush()
    return results


def _ekey(e) -> str:
    import hashlib
    return hashlib.sha1(f"{e['model']}|{e['dataset']}|{e['sample_id']}".encode()).hexdigest()[:16]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two judges' label sequences (aligned)."""
    labels = set(a) | set(b)
    n = len(a)
    if not n:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def compare(corpora: dict[str, dict], out_dir: str) -> dict:
    """corpora: {model: {error_key: mode}}. Writes per-judge distributions,
    pairwise kappa, and a majority-vote ensemble."""
    os.makedirs(out_dir, exist_ok=True)
    models = list(corpora)
    keys = sorted(set.intersection(*[set(c) for c in corpora.values()])) if corpora else []

    # pairwise agreement
    kappa = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            a = [corpora[m1][k] for k in keys]
            b = [corpora[m2][k] for k in keys]
            raw = sum(x == y for x, y in zip(a, b)) / len(keys) if keys else 0.0
            kappa[f"{m1} vs {m2}"] = {"raw_agreement": round(raw, 3),
                                      "cohen_kappa": round(cohen_kappa(a, b), 3)}

    # per-judge overall distribution
    dist = {m: dict(Counter(corpora[m][k] for k in keys)) for m in models}

    # majority-vote ensemble per error
    ensemble = {}
    for k in keys:
        votes = Counter(corpora[m][k] for m in models)
        ensemble[k] = votes.most_common(1)[0][0]

    summary = {"n_common_errors": len(keys), "judges": models,
               "pairwise_agreement": kappa,
               "per_judge_distribution": dist,
               "ensemble_distribution": dict(Counter(ensemble.values()))}
    with open(os.path.join(out_dir, "multijudge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
