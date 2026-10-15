"""Dependency-free lexical vectors for small investigation evidence packets."""

from __future__ import annotations

import math
import re
import hashlib
from collections import Counter

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def _normalize_token(token: str) -> str:
    token = token.lower()
    families = {
        "escalat": "escalate", "investigat": "investigate", "evidenc": "evidence",
        "transact": "transaction", "polic": "policy", "collect": "collect",
        "summar": "summary", "flag": "flag",
    }
    for prefix, canonical in families.items():
        if token.startswith(prefix):
            return canonical
    return token


def embed_text(text: str) -> Counter[str]:
    return Counter(_normalize_token(token) for token in TOKEN_PATTERN.findall(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    """Create a deterministic local embedding for offline knowledge retrieval."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in (_normalize_token(token) for token in TOKEN_PATTERN.findall(text)):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest, "little") % dimensions] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector.tolist()


def vector_cosine(left: list[float], right: list[float]) -> float:
    return float(np.dot(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))
