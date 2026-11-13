"""Nearest-neighbour example selection for the few-shot baseline (TF-IDF cosine similarity, numpy only)."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from lora_ft.data import Example

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TfidfRetriever:
    """Selects the k training examples most similar to a query.

    Retrieved demonstrations are a much stronger few-shot baseline than a fixed random set when there are 77
    classes, because the relevant labels are likely to appear among the neighbours.
    """

    def __init__(self, pool: list[Example]):
        self.pool = pool
        docs = [tokenize(ex.text) for ex in pool]
        df = Counter(term for doc in docs for term in set(doc))
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)], dtype=np.float32)
        self.matrix = np.stack([self._vector(doc) for doc in docs]) if docs else np.zeros((0, len(df)))

    def _vector(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for term, count in Counter(tokens).items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = count
        vec *= self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def top_k(self, text: str, k: int) -> list[Example]:
        """Most similar first in the returned list; callers reverse it so the closest example sits next to the query."""
        if k <= 0 or not self.pool:
            return []
        scores = self.matrix @ self._vector(tokenize(text))
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.pool[i] for i in order]
