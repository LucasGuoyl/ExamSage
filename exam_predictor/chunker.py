"""Chunk normalization layer.

Takes raw (text, metadata) units from ingest.py and splits them into
fixed-size Chunks with overlap. Optionally supports LLM-based semantic
chunking (expensive but better for cross-paragraph concepts).
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from .schema import Chunk


def _hash_id(text: str, prefix: str = "") -> str:
    """Generate a short stable hash id for a chunk."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}" if prefix else h


def _sliding_window(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping windows. Tries to break at sentence boundaries.

    Works for both English (space-separated) and Chinese (no spaces).
    Uses Unicode-aware sentence-ish boundaries: . ! ? 。 ! ? ; ;
    """
    if len(text) <= chunk_size:
        return [text]

    # find candidate break points
    break_re = re.compile(r"[.!?。!?;;\n]+")
    breaks = [m.end() for m in break_re.finditer(text)]
    breaks.append(len(text))

    out: list[str] = []
    start = 0
    while start < len(text):
        target_end = start + chunk_size
        # find the rightmost break <= target_end
        end = target_end
        for b in breaks:
            if b > target_end:
                break
            if b > start:
                end = b
        if end <= start:
            end = min(start + chunk_size, len(text))
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def chunk_units(
    units: Iterable[tuple[str, dict]],
    chunk_size: int = 800,
    overlap: int = 100,
    id_prefix: str = "chunk",
) -> list[Chunk]:
    """Convert raw ingest output into a list of Chunks.

    Each unit (one PDF page / one PPT slide / one MD section) is split into
    one or more Chunks via sliding window if it exceeds chunk_size.
    """
    chunks: list[Chunk] = []
    for unit_idx, (text, meta) in enumerate(units):
        pieces = _sliding_window(text, chunk_size, overlap)
        for piece_idx, piece in enumerate(pieces):
            chunk_meta = dict(meta)
            chunk_meta["unit_idx"] = unit_idx
            chunk_meta["piece_idx"] = piece_idx
            chunk_meta["n_pieces"] = len(pieces)
            chunks.append(Chunk(
                id=_hash_id(piece, prefix=id_prefix),
                source=meta.get("source", "unknown"),
                text=piece,
                metadata=chunk_meta,
            ))
    return chunks


def questions_from_records(records: list[dict]) -> list:
    """Convert raw dicts from ingest.discover_past_papers to ExamQuestion objects."""
    from .schema import ExamQuestion
    out = []
    for r in records:
        if not r.get("text"):
            continue
        out.append(ExamQuestion(
            id=r.get("id") or _hash_id(r["text"], "q"),
            text=r["text"],
            year=r.get("year"),
            paper=r.get("paper"),
            answer=r.get("answer"),
            question_type=r.get("type") or r.get("question_type"),
            difficulty=r.get("difficulty"),
        ))
    return out
