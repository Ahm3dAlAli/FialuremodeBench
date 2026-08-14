"""Recover VQA/diagram images the eval never saved (AI2D, SEED, MME, CRPE,
RealWorldQA) so the human-study page shows them. VLMEvalKit TSVs carry the image
as a base64 column keyed by `index`; our sample_id is `<dataset>-<index>`.

Run ON ROLF (needs LMUData). Reads results/<run>/images_needed.txt, extracts the
missing VQA images into results/<run>/predictions/images/ using the same naming
the study expects (<model>-<dataset>-<index>.png). Recognition names are skipped
(already present). Idempotent.

    python scripts/extract_vqa_images.py --run main2b \
        --lmudata /local/scratch/alali/LMUData
"""
import argparse
import base64
import glob
import io
import os
import re
import sys

# dataset key -> VLMEvalKit TSV stem
TSV = {"ai2d": "AI2D_TEST", "mme": "MME", "seed": "SEEDBench_IMG",
       "realworldqa": "RealWorldQA", "crpe": "CRPE_RELATION"}


def load_index(tsv_path):
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t")
    return {str(i): v for i, v in zip(df["index"], df["image"])}


def load_paths(tsv_path):
    """index(str) -> image_path (relative file), for TSVs that reference unpacked
    image files (e.g. MME) instead of embedding base64."""
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t")
    if "image_path" not in df.columns:
        return {}
    return {str(i): str(p) for i, p in zip(df["index"], df["image_path"])}


def save_file(src, out):
    from PIL import Image
    Image.open(src).convert("RGB").save(out)


def load_choices(tsv_path):
    """index(str) -> {"options": {A:txt,...}, "answer": raw, "answer_text": resolved}.
    Only meaningful for multiple-choice sets with A/B/C/D columns."""
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t")
    letters = [c for c in ["A", "B", "C", "D", "E"] if c in df.columns]
    out = {}
    if not letters:
        return out
    for _, r in df.iterrows():
        opts = {L: str(r[L]) for L in letters if str(r.get(L, "nan")) != "nan"}
        ans = str(r.get("answer", ""))
        atext = opts.get(ans.strip(), "") if ans.strip() in opts else ""
        out[str(r["index"])] = {"options": opts, "answer": ans, "answer_text": atext}
    return out


def save_b64(b64, out):
    from PIL import Image
    raw = base64.b64decode(b64)
    Image.open(io.BytesIO(raw)).convert("RGB").save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--lmudata", default="/local/scratch/alali/LMUData")
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(here, "results", a.run)
    dest = os.path.join(run_dir, "predictions", "images")
    os.makedirs(dest, exist_ok=True)
    needed = [l.strip() for l in open(os.path.join(run_dir, "images_needed.txt")) if l.strip()]

    # group needed VQA names by dataset: <model>-<dataset>-<index>.png. Include
    # rows whose image already exists -- we still need their choices sidecar.
    want = {}  # dataset -> list of (index, out_basename)
    for name in needed:
        m = re.match(r"(.+?)-(" + "|".join(TSV) + r")-(.+)\.png$", name)
        if not m:
            continue
        _, ds, idx = m.groups()
        want.setdefault(ds, []).append((idx, name))

    if not want:
        print("nothing to extract (all present or none are VQA)")
        return
    total = ok = 0
    import json
    choices = {}  # sample_id "<dataset>-<index>" -> {options, answer, answer_text}
    for ds, items in sorted(want.items()):
        cand = glob.glob(os.path.join(a.lmudata, TSV[ds] + "*.tsv"))
        if not cand:
            print(f"{ds:12} NO TSV under {a.lmudata} (stem {TSV[ds]})")
            continue
        idxmap = load_index(cand[0])
        pathmap = load_paths(cand[0])
        imgroot = os.path.join(a.lmudata, "images", TSV[ds])
        chmap = load_choices(cand[0])
        got = nch = 0
        for idx, name in items:
            total += 1
            if idx in chmap and chmap[idx]["options"]:
                choices[f"{ds}-{idx}"] = chmap[idx]; nch += 1
            if os.path.exists(os.path.join(dest, name)):
                continue  # image already present; only needed its choices
            out = os.path.join(dest, name)
            b64 = idxmap.get(idx)
            try:
                # base64-embedded (AI2D/SEED/...) or a file reference (MME)
                if isinstance(b64, str) and len(b64) > 64:
                    save_b64(b64, out); got += 1; ok += 1
                elif idx in pathmap:
                    src = os.path.join(imgroot, pathmap[idx])
                    if os.path.exists(src):
                        save_file(src, out); got += 1; ok += 1
                    else:
                        print(f"  {name}: file not found {src}")
            except Exception as e:
                print(f"  {name}: {e}")
        print(f"{ds:12} {got}/{len(items)} images, {nch} with choices  ({os.path.basename(cand[0])})")
    cj = os.path.join(os.path.dirname(dest), "..", "choices.json")
    json.dump(choices, open(os.path.normpath(cj), "w"))
    print(f"\nextracted {ok}/{total} VQA images into {dest}")
    print(f"wrote {len(choices)} choice-sets -> {os.path.normpath(cj)}")


if __name__ == "__main__":
    main()
