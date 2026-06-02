"""Minimal vector store backed by FAISS (CPU).

Used for two purposes:
  1. Indexing chunk embeddings; query by question to find supporting chunks
  2. Indexing past-question embeddings; query by tutorial to find overlap

This is intentionally simple — no metadata filtering, no persistence
beyond a save/load. Easy to swap for Qdrant / Chroma later if needed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Generic, TypeVar

import faiss
import numpy as np


T = TypeVar("T")


class FAISSStore(Generic[T]):
    """Wraps a FAISS inner-product index over L2-normalized embeddings.

    With normalized vectors, inner product == cosine similarity, so all
    similarity scores fall in [-1, 1] (usually [0, 1] for natural text).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._items: list[T] = []

    def add(self, embeddings: np.ndarray, items: list[T]) -> None:
        """Add a batch of (embedding, item) pairs."""
        if len(embeddings) != len(items):
            raise ValueError("embeddings and items must have same length")
        if len(embeddings) == 0:
            return
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self._items.extend(items)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[list[tuple[T, float]]]:
        """For each query, return up to top_k (item, similarity) pairs."""
        if len(self._items) == 0:
            return [[] for _ in range(len(query_embeddings))]
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        k = min(top_k, len(self._items))
        scores, idxs = self.index.search(query_embeddings, k)
        results: list[list[tuple[T, float]]] = []
        for row_scores, row_idxs in zip(scores, idxs):
            row: list[tuple[T, float]] = []
            for s, i in zip(row_scores, row_idxs):
                if i == -1:
                    continue
                row.append((self._items[i], float(s)))
            results.append(row)
        return results

    def __len__(self) -> int:
        return len(self._items)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump({"dim": self.dim, "items": self._items}, f)

    @classmethod
    def load(cls, path: str | Path) -> "FAISSStore":
        path = Path(path)
        with open(path.with_suffix(".pkl"), "rb") as f:
            data = pickle.load(f)
        store = cls(dim=data["dim"])
        store.index = faiss.read_index(str(path.with_suffix(".faiss")))
        store._items = data["items"]
        return store
