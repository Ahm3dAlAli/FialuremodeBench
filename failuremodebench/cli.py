"""FailureModeBench command-line driver.

Subcommands (run in order for a full evaluation):

  infer-recognition   run VLM classification over recognition datasets
  infer-vqa           run VLMEvalKit over VQA datasets, import predictions
  judge               extract errors + LLM-judge them into F1-F8
  aggregate           build failure-rate tables, confusion matrices, figures
  all                 judge + aggregate (assumes predictions already present)

Everything writes under a single run dir (default: results/<run>).
"""
from __future__ import annotations

import argparse
import json
import os

from . import aggregate, figures
from .config import (DATASETS, MODELS, DEFAULT_DATASETS, DEFAULT_MODELS,
                     datasets_by_modality)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _list(arg: str, default: list) -> list:
    vals = [x for x in (arg or "").split(",") if x]
    return vals or default


def _run_dir(args) -> str:
    d = os.path.join(_repo_root(), "results", args.run, "predictions")
    os.makedirs(d, exist_ok=True)
    return d


def cmd_infer_recognition(args):
    from .backends import build_backend
    from .recognition import run_recognition
    run_dir = _run_dir(args)
    models = _list(args.models, DEFAULT_MODELS)
    dsets = [k for k in _list(args.datasets, DEFAULT_DATASETS)
             if DATASETS[k].modality == "recognition"]
    stats = []
    for mk in models:
        backend, _ = build_backend(mk, prefer_api=args.api, api_provider=args.provider)
        for dk in dsets:
            print(f"[recognition] {mk} x {dk}")
            try:
                s = run_recognition(DATASETS[dk], backend, mk, run_dir, _repo_root(),
                                    limit=args.limit, use_embeddings=args.embeddings,
                                    label_hint_k=args.label_hint_k,
                                    shuffle_labels=args.shuffle_labels,
                                    hint_distractors=args.hint_distractors)
                print("   ", s)
                stats.append(s)
            except Exception as e:
                import traceback
                print(f"   !! FAILED {mk} x {dk}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(json.dumps(stats, indent=2))


def cmd_infer_clip(args):
    """Zero-shot CLIP recognition (contrastive baseline) over recognition datasets."""
    from .backends import build_backend
    from .recognition import run_recognition_clip
    run_dir = _run_dir(args)
    models = [m for m in _list(args.models, DEFAULT_MODELS) if MODELS[m].family == "clip"]
    dsets = [k for k in _list(args.datasets, DEFAULT_DATASETS)
             if DATASETS[k].modality == "recognition"]
    stats = []
    for mk in models:
        print(f"[clip] loading {mk} ...")
        backend, _ = build_backend(mk)
        print(f"[clip] {mk} -> open_clip spec {getattr(backend, 'spec_used', '?')}")
        for dk in dsets:
            print(f"[clip] {mk} x {dk}")
            try:
                s = run_recognition_clip(DATASETS[dk], backend, mk, run_dir,
                                         _repo_root(), limit=args.limit)
                print("   ", s); stats.append(s)
            except Exception as e:
                import traceback
                print(f"   !! FAILED {mk} x {dk}: {type(e).__name__}: {e}")
                traceback.print_exc()
        # free the encoder before loading the next model (contended 11GB box)
        del backend
        try:
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
    print(json.dumps(stats, indent=2))


def cmd_infer_vqa(args):
    from .vqa import import_predictions, run_vlmevalkit, _find_pred_file
    run_dir = _run_dir(args)
    work = os.path.join(_repo_root(), "results", args.run)
    models = _list(args.models, DEFAULT_MODELS)
    dsets = [k for k in _list(args.datasets, DEFAULT_DATASETS)
             if DATASETS[k].modality == "vqa"]
    for mk in models:
        for dk in dsets:
            spec = DATASETS[dk]
            print(f"[vqa] {mk} x {dk}")
            try:
                # VLMEvalKit run.py has no --limit; VQA always runs the full set.
                out_dir = run_vlmevalkit(mk, spec, work)
                pred = _find_pred_file(out_dir, spec)
                if not pred:
                    print(f"   !! no prediction file for {mk} x {dk}")
                    continue
                _, s = import_predictions(pred, mk, spec, run_dir)
                print("   ", s)
            except Exception as e:
                import traceback
                print(f"   !! FAILED {mk} x {dk}: {type(e).__name__}: {e}")
                traceback.print_exc()


def cmd_judge(args):
    from .judge import gather_errors, judge_errors, sample_per_family
    from .providers import get_provider
    run_dir = _run_dir(args)
    errors = gather_errors(run_dir)
    print(f"[judge] gathered {len(errors)} errors")
    sampled = sample_per_family(errors, args.n_per_family)
    print(f"[judge] sampled {len(sampled)} across families")
    provider = get_provider("echo" if args.echo else args.provider, args.judge_model)
    corpus = os.path.join(_repo_root(), "results", args.run, "corpus.jsonl")
    counts = judge_errors(sampled, provider, corpus, attach_images=not args.no_images)
    print(f"[judge] verdict counts: {json.dumps(counts)}")
    print(f"[judge] corpus -> {corpus}")


def cmd_aggregate(args):
    run_dir = _run_dir(args)
    base = os.path.join(_repo_root(), "results", args.run)
    out_dir = os.path.join(base, "tables")
    fig_dir = os.path.join(base, "figures")
    corpus = os.path.join(base, "corpus.jsonl")
    summary = aggregate.run_all(run_dir, corpus, out_dir)
    figs = figures.make_all(out_dir, fig_dir)
    summary["figures"] = figs
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def cmd_all(args):
    cmd_judge(args)
    cmd_aggregate(args)


def main(argv=None):
    p = argparse.ArgumentParser("failuremodebench")
    p.add_argument("--run", default="main", help="run name (results/<run>/)")
    # comma-separated (NOT nargs) so a trailing subcommand is never greedily eaten
    p.add_argument("--models", default="", help="comma-separated model keys")
    p.add_argument("--datasets", default="", help="comma-separated dataset keys")
    p.add_argument("--limit", type=int, default=None, help="cap samples/dataset")
    p.add_argument("--provider", default="anthropic", help="judge/API-VLM provider")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--n-per-family", type=int, default=200)
    p.add_argument("--api", action="store_true", help="recognition via API VLM")
    p.add_argument("--embeddings", action="store_true", help="semantic label match")
    p.add_argument("--label-hint-k", type=int, default=0,
                   help="closed-set control: list the candidate labels in the prompt "
                        "when a dataset has <= K classes (0 = open-ended, the default)")
    p.add_argument("--shuffle-labels", type=int, default=0,
                   help="permute the closed-set label order with this seed "
                        "(ordering-bias robustness check; 0 = fixed order)")
    p.add_argument("--hint-distractors", type=int, default=0,
                   help="large-label closed-set: prompt with gold + K random "
                        "distractors per item (tests F7 collapse at large label spaces)")
    p.add_argument("--no-images", action="store_true", help="text-only judge")
    p.add_argument("--echo", action="store_true", help="offline stub judge (tests)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("infer-recognition", cmd_infer_recognition),
                     ("infer-clip", cmd_infer_clip),
                     ("infer-vqa", cmd_infer_vqa), ("judge", cmd_judge),
                     ("aggregate", cmd_aggregate), ("all", cmd_all)]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
