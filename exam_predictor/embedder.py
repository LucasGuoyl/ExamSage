"""Cloud-only embedding adapter.

ExamSage deliberately does not download or execute local AI models. Local code
only normalizes vectors returned by the user's selected cloud provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2-normalized float32 matrix with shape (N, dim)."""


class ProviderEmbedder(Embedder):
    def __init__(self, provider, batch_size: int = 96):
        self.provider = provider
        self.batch_size = batch_size
        self.dim = int(provider.embedding_dimension)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches: list[np.ndarray] = []
        for index in range(0, len(texts), self.batch_size):
            batch = list(texts[index:index + self.batch_size])
            vectors = np.asarray(self.provider.embed(batch), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape != (len(batch), self.dim):
                raise ValueError(
                    f"Provider returned embedding shape {vectors.shape}; "
                    f"expected ({len(batch)}, {self.dim})."
                )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            batches.append(vectors / norms)
        return np.vstack(batches)


def build_embedder(config: dict, provider=None) -> Embedder:
    embedding_config = config.get("embedding", {})
    backend = embedding_config.get("backend", "provider")
    if backend != "provider":
        raise ValueError(
            "ExamSage supports cloud provider embeddings only. "
            "Set embedding.backend to 'provider'."
        )
    if provider is None:
        raise ValueError("embedding.backend=provider requires an AI provider")
    return ProviderEmbedder(
        provider=provider,
        batch_size=int(embedding_config.get("batch_size", 96)),
    )
