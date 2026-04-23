"""Utility helpers."""

from __future__ import annotations

import hashlib


def stable_hash_to_unit_interval(text: str) -> float:
    """Map text to a stable number in [0, 1) using SHA-256."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    integer_value = int(digest, 16)
    return integer_value / float(16 ** len(digest))


def server_score(task_type: str, server_id: int) -> float:
    """Stable deterministic score for a task/server pair."""
    return stable_hash_to_unit_interval(f"{task_type}:{server_id}")

