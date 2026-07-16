"""Small NumPy cosine-similarity store for local deterministic retrieval.

The vectors themselves come from the selected cloud provider. NumPy only
compares those numbers; no local AI model is downloaded or executed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np


T = TypeVar("T")


class FAISSStore(Generic[T]):
    """Compatibility name for the former FAISS-backed normalized-vector store."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._items: list[T] = []

    def add(self, embeddings: np.ndarray, items: list[T]) -> None:
        if len(embeddings) != len(items):
            raise ValueError("embeddings and items must have same length")
        if len(embeddings) == 0:
            return
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected (*, {self.dim}) embeddings, got {vectors.shape}")
        self._vectors = np.vstack([self._vectors, vectors])
        self._items.extend(items)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[list[tuple[T, float]]]:
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"Expected (*, {self.dim}) queries, got {queries.shape}")
        if not self._items:
            return [[] for _ in range(len(queries))]
        k = min(max(0, top_k), len(self._items))
        if k == 0:
            return [[] for _ in range(len(queries))]
        similarities = queries @ self._vectors.T
        # Argpartition avoids a full O(n log n) sort, then the small selected
        # subset is ordered deterministically by similarity.
        selected = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        results: list[list[tuple[T, float]]] = []
        for row_index, indices in enumerate(selected):
            ordered = indices[np.argsort(-similarities[row_index, indices], kind="stable")]
            results.append([
                (self._items[int(index)], float(similarities[row_index, index]))
                for index in ordered
            ])
        return results

    def __len__(self) -> int:
        return len(self._items)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self._vectors, allow_pickle=False)
        with path.with_suffix(".pkl").open("wb") as stream:
            pickle.dump({"dim": self.dim, "items": self._items}, stream)

    @classmethod
    def load(cls, path: str | Path) -> "FAISSStore":
        path = Path(path)
        with path.with_suffix(".pkl").open("rb") as stream:
            data = pickle.load(stream)  # trusted local ExamSage state only
        store = cls(dim=int(data["dim"]))
        store._vectors = np.load(path.with_suffix(".npy"), allow_pickle=False)
        store._items = data["items"]
        return store
