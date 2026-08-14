"""Prefetch every FailureModeBench dataset onto rolf, under the user's scratch.

Checks existence first (HF cache / VLMEvalKit LMUData / local shared mirror) and
only downloads what is missing. Nothing touches the home quota:

  HF_HOME  -> /local/scratch/<user>/hf_cache   (recognition datasets)
  LMUData  -> /local/scratch/<user>/LMUData    (VLMEvalKit VQA datasets)
  ImageNet -> reused from the shared mirror if present (no gated download)

Usage (on rolf, inside the fmb env):
    export HF_HOME=/local/scratch/$USER/hf_cache
    export LMUData=/local/scratch/$USER/LMUData
    python scripts/prefetch_datasets.py            # all
    python scripts/prefetch_datasets.py --only recognition
    python scripts/prefetch_datasets.py --only vqa
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from failuremodebench.config import DATASETS  # noqa: E402

SHARED_IMAGENET_VAL = "/local/scratch/datasets/ImageNet/ILSVRC2012/val"


def human(n):
    for u in ["B", "K", "M", "G", "T"]:
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}P"


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def prefetch_recognition(specs):
    from datasets import load_dataset
    for s in specs:
        if s.key == "imagenet" and os.path.isdir(SHARED_IMAGENET_VAL):
            print(f"[recognition] {s.key:16} REUSE shared mirror {SHARED_IMAGENET_VAL}")
            continue
        try:
            print(f"[recognition] {s.key:16} fetching {s.hf_path} "
                  f"({s.hf_config or ''} {s.hf_split}) ...", flush=True)
            kw = dict(split=s.hf_split, verification_mode="no_checks")
            if getattr(s, "trust_remote_code", False):
                kw["trust_remote_code"] = True
            ds = load_dataset(s.hf_path, s.hf_config, **kw) if s.hf_config \
                else load_dataset(s.hf_path, **kw)
            print(f"[recognition] {s.key:16} OK  n={ds.num_rows}")
        except Exception as e:
            print(f"[recognition] {s.key:16} FAIL {type(e).__name__}: {e}")


def prefetch_vqa(specs):
    try:
        from vlmeval.dataset import build_dataset
    except Exception as e:
        print(f"[vqa] VLMEvalKit not importable ({e}); skip. Fix env first.")
        return
    for s in specs:
        name = s.vlmevalkit_name
        try:
            print(f"[vqa] {s.key:16} building {name} ...", flush=True)
            d = build_dataset(name)
            n = len(d.data) if d is not None and hasattr(d, "data") else "?"
            print(f"[vqa] {s.key:16} OK  ({name}) n={n}")
        except Exception as e:
            print(f"[vqa] {s.key:16} FAIL {name}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["recognition", "vqa"], default=None)
    args = ap.parse_args()
    print("HF_HOME =", os.environ.get("HF_HOME", "(unset!)"))
    print("LMUData =", os.environ.get("LMUData", "(unset!)"))
    recog = [d for d in DATASETS.values() if d.modality == "recognition"]
    vqa = [d for d in DATASETS.values() if d.modality == "vqa"]
    if args.only in (None, "recognition"):
        prefetch_recognition(recog)
    if args.only in (None, "vqa"):
        prefetch_vqa(vqa)
    hf = os.environ.get("HF_HOME", "")
    if hf and os.path.isdir(hf):
        print(f"\nHF cache size: {human(dir_size(hf))} at {hf}")
    lmu = os.environ.get("LMUData", "")
    if lmu and os.path.isdir(lmu):
        print(f"LMUData size:  {human(dir_size(lmu))} at {lmu}")
    print("PREFETCH_DONE")


if __name__ == "__main__":
    main()
