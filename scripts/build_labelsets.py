"""Generate failuremodebench/labelsets/*.json from HF dataset metadata.

Pulls ClassLabel names via load_dataset_builder (metadata only, no bulk
download). ImageNet-A/-R are 200-class subsets, so they get their own files.
Run once (needs network to HuggingFace):  python scripts/build_labelsets.py
"""
import json
import os
import sys
import urllib.request

from datasets import load_dataset, load_dataset_builder

IN1K_URL = ("https://raw.githubusercontent.com/anishathalye/"
            "imagenet-simple-labels/master/imagenet-simple-labels.json")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "failuremodebench", "labelsets")
os.makedirs(OUT, exist_ok=True)

# ClassLabel-based sources (metadata only).
# (labelset_name, hf_path, hf_config, label_column)
SOURCES = [
    ("food101", "food101", None, "label"),
    ("dtd", "tanganke/dtd", None, "label"),
    ("resisc45", "tanganke/resisc45", None, "label"),
    ("imagenet_a", "barkermrl/imagenet-a", None, "label"),
]


def write_labelset(name, labels):
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"labels": labels, "synonyms": {}}, f, indent=0)


def build_imagenet():
    """1000 public simple labels (imagenet-1k on the Hub is gated)."""
    with urllib.request.urlopen(IN1K_URL, timeout=60) as r:
        labels = json.load(r)
    assert len(labels) == 1000, len(labels)
    write_labelset("imagenet", labels)
    return f"imagenet: {len(labels)} labels (public list)"


def build_imagenet_r():
    """axiong/imagenet-r stores gold as a `class_name` string; collect the 200
    unique names by streaming (no full download)."""
    ds = load_dataset("axiong/imagenet-r", split="test", streaming=True)
    seen, order = set(), []
    for i, ex in enumerate(ds):
        cn = ex.get("class_name")
        if cn and cn not in seen:
            seen.add(cn)
            order.append(clean(cn))
        if len(seen) >= 200 or i > 60000:
            break
    write_labelset("imagenet_r", sorted(order))
    return f"imagenet_r: {len(order)} labels (streamed uniques)"


def names_from_builder(path, config, col):
    b = load_dataset_builder(path, config) if config else load_dataset_builder(path)
    feat = b.info.features
    if feat and col in feat and hasattr(feat[col], "names"):
        return list(feat[col].names)
    raise RuntimeError(f"{path}: no ClassLabel names on column {col!r}")


def clean(name: str) -> str:
    # ImageNet synset labels look like 'n01440764' + ', tench, Tinca tinca'
    return name.replace("_", " ").strip()


def main():
    ok, fail = [], []
    for lsname, path, config, col in SOURCES:
        try:
            names = [clean(n) for n in names_from_builder(path, config, col)]
            write_labelset(lsname, names)
            ok.append(f"{lsname}: {len(names)} labels")
        except Exception as e:
            fail.append(f"{lsname} ({path}): {type(e).__name__}: {e}")
    for name, fn in [("imagenet", build_imagenet), ("imagenet_r", build_imagenet_r)]:
        try:
            ok.append(fn())
        except Exception as e:
            fail.append(f"{name}: {type(e).__name__}: {e}")
    print("OK:")
    for x in ok:
        print("  ", x)
    if fail:
        print("FAILED (fix path in config.py or add a manual list):")
        for x in fail:
            print("  ", x)
        sys.exit(1 if not ok else 0)


if __name__ == "__main__":
    main()
