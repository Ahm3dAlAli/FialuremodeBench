"""Map a VLM's free-text answer onto a fixed classification label set.

Generative VLMs answer ImageNet/Food101/... in prose ("This looks like a golden
retriever dog"). To score top-1 accuracy we must resolve that prose to one of
the N canonical labels. Strategy, cheapest first:

  1. exact normalised match / label appears as a substring of the answer;
  2. answer appears as a substring of a label (or a known synonym);
  3. token-overlap (Jaccard) best label above a floor;
  4. optional semantic match via sentence-transformers embeddings (if
     installed and enabled) -- this is what production CLIP-benchmark harnesses
     use and is recommended on rolf.

Steps 1-3 are pure-python and unit-tested offline. Step 4 is opt-in.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_/]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> set[str]:
    return set(normalize(s).split())


class LabelMatcher:
    def __init__(self, labels: list[str], synonyms: Optional[dict[str, list[str]]] = None,
                 use_embeddings: bool = False, embed_model: str = "all-MiniLM-L6-v2"):
        self.labels = labels
        self.norm_labels = [normalize(x) for x in labels]
        self.label_tokens = [_tokens(x) for x in labels]
        self.synonyms = {normalize(k): [normalize(v) for v in vs]
                         for k, vs in (synonyms or {}).items()}
        self.use_embeddings = use_embeddings
        self._embedder = None
        self._label_emb = None
        if use_embeddings:
            self._init_embeddings(embed_model)

    def _init_embeddings(self, embed_model):
        from sentence_transformers import SentenceTransformer  # heavy; opt-in
        self._embedder = SentenceTransformer(embed_model)
        self._label_emb = self._embedder.encode(
            self.labels, convert_to_numpy=True, normalize_embeddings=True)

    def match(self, answer: str) -> tuple[Optional[int], str]:
        """Return (label_index or None, method). None => unmatched."""
        a = normalize(answer)
        if not a:
            return None, "empty"

        # 1. label substring of answer (prefer the longest label that fits)
        hits = [(i, nl) for i, nl in enumerate(self.norm_labels)
                if nl and re.search(rf"\b{re.escape(nl)}\b", a)]
        if hits:
            hits.sort(key=lambda t: len(t[1]), reverse=True)
            return hits[0][0], "substring"

        # 2. synonyms
        for i, lbl in enumerate(self.norm_labels):
            for syn in self.synonyms.get(lbl, []):
                if syn and re.search(rf"\b{re.escape(syn)}\b", a):
                    return i, "synonym"

        # 3. token Jaccard
        at = _tokens(answer)
        best_i, best_j = None, 0.0
        for i, lt in enumerate(self.label_tokens):
            if not lt:
                continue
            j = len(at & lt) / len(at | lt)
            if j > best_j:
                best_i, best_j = i, j
        if best_i is not None and best_j >= 0.34:
            return best_i, f"jaccard:{best_j:.2f}"

        # 4. embeddings (opt-in)
        if self.use_embeddings and self._embedder is not None:
            import numpy as np
            q = self._embedder.encode([answer], convert_to_numpy=True,
                                      normalize_embeddings=True)[0]
            sims = self._label_emb @ q
            i = int(sims.argmax())
            if sims[i] >= 0.35:
                return i, f"embed:{float(sims[i]):.2f}"

        return None, "unmatched"


@lru_cache(maxsize=None)
def classification_prompt(n_labels: int, label_hint: str = "") -> str:
    """Prompt shown to the VLM for a recognition task."""
    base = ("Identify the main subject of this image. "
            "Answer with the single most specific category name only, "
            "no explanation.")
    if label_hint:
        base += f" Choose from these categories: {label_hint}."
    return base
