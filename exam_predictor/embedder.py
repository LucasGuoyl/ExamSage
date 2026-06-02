"""Embedding backends.

Two interchangeable implementations:
  - LocalEmbedder: uses sentence-transformers (BGE recommended for Chinese)
  - APIEmbedder: uses any OpenAI-compatible embedding endpoint

Both expose `embed(texts: list[str]) -> np.ndarray` with shape (N, D).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential


class Embedder(ABC):
    """Abstract embedding backend."""

    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return shape (N, dim) float32 array. L2-normalized."""
        ...


class LocalEmbedder(Embedder):
    """Sentence-Transformers backend (default: BGE for Chinese)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        # peek at dim
        test = self.model.encode(["test"], convert_to_numpy=True)
        self.dim = int(test.shape[1])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return embeddings.astype(np.float32)


class APIEmbedder(Embedder):
    """OpenAI-compatible embedding API backend."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 128,
        dim: int | None = None,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = batch_size
        # for known models we hard-code the dim; otherwise discover by a test call
        known_dims = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        if dim is not None:
            self.dim = dim
        elif model in known_dims:
            self.dim = known_dims[model]
        else:
            resp = self.client.embeddings.create(model=model, input=["test"])
            self.dim = len(resp.data[0].embedding)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=batch)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # L2-normalize (some endpoints don't)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            out.append(self._embed_batch(batch))
        return np.vstack(out)


def build_embedder(config: dict) -> Embedder:
    """Factory: instantiate Embedder from a config dict."""
    emb_cfg = config.get("embedding", {})
    backend = emb_cfg.get("backend", "local")
    if backend == "local":
        return LocalEmbedder(
            model_name=emb_cfg.get("model_name", "BAAI/bge-large-zh-v1.5"),
            device=emb_cfg.get("device", "cpu"),
            batch_size=emb_cfg.get("batch_size", 32),
        )
    elif backend == "api":
        llm_cfg = config.get("llm", {})
        return APIEmbedder(
            model=emb_cfg.get("api_model", "text-embedding-3-large"),
            api_key=emb_cfg.get("api_key", llm_cfg.get("api_key")),
            base_url=emb_cfg.get("base_url", llm_cfg.get("base_url")),
            batch_size=emb_cfg.get("batch_size", 128),
        )
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
