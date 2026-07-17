"""Automated Failure Discovery for VLMs.

Where the F1-F8 judge *labels* errors into a fixed taxonomy, this module
*discovers* failure modes bottom-up and actively confirms them:

  1. EMBED    each error (dataset|question|gold->prediction, optionally the image)
              into a vector.
  2. CLUSTER  the errors into coherent "failure slices" (KMeans w/ silhouette
              model-selection, or HDBSCAN if installed) -- no taxonomy assumed.
  3. DESCRIBE each slice with the LLM: a name, a one-line pattern description, the
              closest F1-F8 mode (or "novel" if it fits none), and a severity.
  4. AMPLIFY  (active step) ask the LLM to propose fresh probe cases that should
              trigger the slice's pattern, run them through a VLM backend, and
              measure the reproduce-rate -- confirming the failure is real and
              test-set-independent, not a labeling artifact.
  5. RANK     slices by prevalence x coherence x severity into a report.

Steps 1-3 + ranking are GPU-free and unit-tested offline; step 4 uses a VLM
backend (rolf) and is optional.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import taxonomy
from .records import read_jsonl, write_jsonl


# --------------------------------------------------------------------------- #
# 1. Embedding                                                                 #
# --------------------------------------------------------------------------- #
def error_text(e: dict) -> str:
    return (f"dataset={e.get('dataset')} family={e.get('family')} | "
            f"Q: {e.get('question','')[:400]} | "
            f"gold: {e.get('gold','')[:120]} | pred: {e.get('prediction','')[:200]}")


def embed_errors(errors: list[dict], method: str = "text",
                 st_model: str = "all-MiniLM-L6-v2"):
    """Return an (N, D) numpy array. method: text | tfidf | clip."""
    import numpy as np
    texts = [error_text(e) for e in errors]
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        X = TfidfVectorizer(max_features=4096).fit_transform(texts)
        d = min(128, X.shape[1] - 1) if X.shape[1] > 2 else X.shape[1]
        return TruncatedSVD(n_components=d).fit_transform(X).astype("float32")
    if method == "clip":
        return _embed_clip(errors)
    # default: sentence-transformers on the textual rendering
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(st_model)
    return m.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False).astype("float32")


def _embed_clip(errors: list[dict]):
    """Joint CLIP image+text embedding (rolf; needs image_ref on errors)."""
    import numpy as np
    import open_clip
    import torch
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    vecs = []
    with torch.no_grad():
        for e in errors:
            t = model.encode_text(tok([error_text(e)[:300]]))
            t = t / t.norm(dim=-1, keepdim=True)
            ref = e.get("image_ref", "")
            if ref and os.path.exists(ref):
                img = preprocess(Image.open(ref).convert("RGB")).unsqueeze(0)
                v = model.encode_image(img)
                v = v / v.norm(dim=-1, keepdim=True)
                vecs.append(((t + v) / 2).squeeze(0).cpu().numpy())
            else:
                vecs.append(t.squeeze(0).cpu().numpy())
    return np.vstack(vecs).astype("float32")


# --------------------------------------------------------------------------- #
# 2. Clustering                                                                #
# --------------------------------------------------------------------------- #
def cluster_errors(X, method: str = "kmeans", k_range=(6, 20), min_size=5):
    """Return an array of cluster labels (-1 = noise). Picks k by silhouette."""
    import numpy as np
    n = len(X)
    if n < min_size * 2:
        return np.zeros(n, dtype=int)
    if method == "hdbscan":
        try:
            import hdbscan
            return hdbscan.HDBSCAN(min_cluster_size=min_size).fit_predict(X)
        except Exception:
            method = "kmeans"
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k, best_s, best_labels = None, -1.0, None
    kmax = min(k_range[1], max(2, n // min_size))
    for k in range(max(2, k_range[0]), kmax + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if s > best_s:
            best_k, best_s, best_labels = k, s, labels
    return best_labels if best_labels is not None else \
        KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)


def _coherence(X, labels):
    """Mean intra-cluster cosine similarity per cluster label."""
    import numpy as np
    out = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            out[c] = 0.0
            continue
        sub = X[idx]
        sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
        sims = sub @ sub.T
        n = len(idx)
        out[c] = float((sims.sum() - n) / (n * (n - 1)))
    return out


# --------------------------------------------------------------------------- #
# 3. Describe slices with the LLM                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Slice:
    slice_id: int
    size: int
    prevalence: float
    coherence: float
    top_datasets: list
    top_families: list
    name: str = ""
    description: str = ""
    failure_mode: str = ""       # closest F1-F8 or "novel"
    severity: float = 0.0
    example_ids: list = field(default_factory=list)
    reproduce_rate: Optional[float] = None
    probes: list = field(default_factory=list)


_DESCRIBE_SYS = f"""\
You analyse clusters of vision-language model ERRORS to discover failure modes.
Given a sample of errors that an unsupervised clustering grouped together, output
STRICT JSON describing the SHARED failure pattern:
  {{"name": "<=6 words", "description": "one sentence: what the model \
systematically gets wrong here", "failure_mode": "F1..F8 or novel", \
"severity": 0-1}}
Use this reference taxonomy for the failure_mode field (or "novel" if the pattern \
fits none well):
{taxonomy.rubric_block()}"""


def describe_slice(sample_errors: list[dict], provider) -> dict:
    lines = []
    for e in sample_errors[:12]:
        lines.append(f"- [{e.get('dataset')}] Q:{e.get('question','')[:160]} "
                     f"| gold:{e.get('gold','')[:60]} | pred:{e.get('prediction','')[:100]}")
    user = ("Errors in this cluster:\n" + "\n".join(lines) +
            '\n\nReturn the JSON described above.')
    try:
        reply = provider.complete(_DESCRIBE_SYS, user, image=None,
                                  max_tokens=300, temperature=0.0)
        s, e = reply.find("{"), reply.rfind("}")
        v = json.loads(reply[s:e + 1])
    except Exception as ex:
        return {"name": "undescribed", "description": f"describe error: {ex}",
                "failure_mode": "novel", "severity": 0.0}
    fm = str(v.get("failure_mode", "novel")).upper()
    v["failure_mode"] = fm if fm in taxonomy.CODES else "novel"
    return v


# --------------------------------------------------------------------------- #
# 4. Active amplification (optional, VLM backend)                             #
# --------------------------------------------------------------------------- #
_PROBE_SYS = ("You design probe questions to test whether a vision-language "
              "model exhibits a specific failure pattern. Output STRICT JSON: "
              '{"probes": ["q1", "q2", "q3"]} -- short, self-contained questions '
              "that, on an image matching the pattern's domain, would expose it.")


def propose_probes(sl: Slice, provider, n: int = 3) -> list[str]:
    user = (f"Failure pattern: {sl.name} -- {sl.description}\n"
            f"Domains: {sl.top_datasets}. Propose {n} probe questions.")
    try:
        reply = provider.complete(_PROBE_SYS, user, max_tokens=300, temperature=0.4)
        s, e = reply.find("{"), reply.rfind("}")
        return list(json.loads(reply[s:e + 1]).get("probes", []))[:n]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# 5. Orchestration                                                            #
# --------------------------------------------------------------------------- #
def _top(counter_items, k=3):
    return [x for x, _ in Counter(counter_items).most_common(k)]


def run_discovery(corpus_or_preds: list[dict], out_dir: str, provider,
                  embed_method: str = "text", cluster_method: str = "kmeans",
                  min_size: int = 5, describe: bool = True) -> dict:
    """Full discovery pass over a list of error dicts. Returns a summary."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    errors = [e for e in corpus_or_preds if not e.get("correct", True)] \
        or corpus_or_preds  # accept a raw prediction list or an error corpus
    if len(errors) < min_size * 2:
        raise RuntimeError(f"need >= {min_size*2} errors to discover slices; got {len(errors)}")

    X = embed_errors(errors, method=embed_method)
    labels = cluster_errors(X, method=cluster_method, min_size=min_size)
    coh = _coherence(X, labels)
    total = len(errors)

    slices: list[Slice] = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idx = [i for i, l in enumerate(labels) if l == c]
        members = [errors[i] for i in idx]
        sl = Slice(
            slice_id=int(c), size=len(idx), prevalence=round(len(idx) / total, 4),
            coherence=round(coh.get(c, 0.0), 4),
            top_datasets=_top([m.get("dataset") for m in members]),
            top_families=_top([m.get("family") for m in members]),
            example_ids=[m.get("sample_id") for m in members[:5]])
        if describe:
            d = describe_slice(members, provider)
            sl.name = d.get("name", "")
            sl.description = d.get("description", "")
            sl.failure_mode = d.get("failure_mode", "novel")
            sl.severity = float(d.get("severity", 0.0))
        slices.append(sl)

    # priority = prevalence * coherence * severity (severity=1 if undescribed)
    for sl in slices:
        sev = sl.severity if sl.severity > 0 else 1.0
        sl_priority = sl.prevalence * max(sl.coherence, 0.0) * sev
        sl.__dict__["priority"] = round(sl_priority, 5)
    slices.sort(key=lambda s: s.__dict__["priority"], reverse=True)

    write_jsonl(os.path.join(out_dir, "slices.jsonl"),
                [asdict(s) | {"priority": s.__dict__.get("priority", 0.0)} for s in slices])
    _write_report(slices, out_dir, total)
    mode_hist = Counter(s.failure_mode for s in slices)
    novel = [s for s in slices if s.failure_mode == "novel"]
    return {"n_errors": total, "n_slices": len(slices),
            "discovered_modes": dict(mode_hist),
            "novel_slice_count": len(novel),
            "top_slice": (slices[0].name or f"slice{slices[0].slice_id}") if slices else None,
            "out_dir": out_dir}


def _write_report(slices, out_dir, total):
    import csv
    with open(os.path.join(out_dir, "slices.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice_id", "priority", "size", "prevalence", "coherence",
                    "severity", "failure_mode", "name", "top_datasets", "description"])
        for s in slices:
            w.writerow([s.slice_id, s.__dict__.get("priority", 0.0), s.size,
                        s.prevalence, s.coherence, s.severity, s.failure_mode,
                        s.name, "|".join(map(str, s.top_datasets)), s.description])
    lines = [f"# Discovered failure slices ({len(slices)} from {total} errors)\n"]
    for i, s in enumerate(slices, 1):
        lines.append(
            f"## {i}. {s.name or f'slice {s.slice_id}'}  "
            f"[{s.failure_mode}]  (priority={s.__dict__.get('priority',0):.4f})\n"
            f"- prevalence {s.prevalence:.1%} ({s.size} errors), "
            f"coherence {s.coherence:.2f}, severity {s.severity:.2f}\n"
            f"- domains: {', '.join(map(str, s.top_datasets))}\n"
            f"- {s.description}\n")
    with open(os.path.join(out_dir, "discovery_report.md"), "w") as f:
        f.write("\n".join(lines))
