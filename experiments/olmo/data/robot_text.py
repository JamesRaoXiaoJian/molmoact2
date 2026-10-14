"""Shared task canonicalization for robot training and native inference."""

import re

_PREFIXES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def normalize_robot_task(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip("\"'`“”‘’[](){}").strip()
        for pattern in _PREFIXES:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(".,!?;:,…").rstrip()
        normalized = normalized.rstrip("\"'”’)]}").rstrip()
        normalized = normalized.rstrip(".,!?;:,…").rstrip()
    chunks = [
        chunk.strip() for chunk in re.split(r"[.!?]+", normalized) if chunk.strip()
    ]
    if len(chunks) > 1:
        normalized = "; ".join(chunks)
    return normalized.lower()
