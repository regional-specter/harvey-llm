"""Extract Harvey context-response pairs from parsed dialogue."""

from __future__ import annotations

import re

from configs.dataset_config import HARVEY_ALIASES
from dataset_builder.models import ContextResponsePair, DialogueLine

MIN_CONTEXT_LEN = 5
MIN_RESPONSE_LEN = 3
MAX_CONTEXT_LEN = 500


def is_harvey(speaker: str | None) -> bool:
    if not speaker:
        return False
    normalized = speaker.lower().strip()
    normalized = re.sub(r"[^a-z.\s']", "", normalized)
    return normalized in HARVEY_ALIASES or normalized.startswith("harvey")


def extract_pairs(lines: list[DialogueLine]) -> list[ContextResponsePair]:
    """Build context-response pairs from sequential dialogue.

    For each Harvey line, the context is the immediately preceding non-Harvey line.
    If multiple consecutive non-Harvey lines precede Harvey, they are merged.
    """
    pairs: list[ContextResponsePair] = []
    pending_context: list[str] = []
    pending_speaker: str | None = None
    episode = _episode_from_source(lines[0].source_file) if lines else None

    for line in lines:
        if is_harvey(line.speaker):
            response = line.text.strip()
            if len(response) < MIN_RESPONSE_LEN:
                continue

            context = _build_context(pending_context)
            if not context or len(context) < MIN_CONTEXT_LEN:
                # Harvey monologue / cold open — use a generic prompt
                context = "What do you have to say?"

            pairs.append(
                ContextResponsePair(
                    context=context[:MAX_CONTEXT_LEN],
                    response=response,
                    speaker=pending_speaker,
                    source="transcript",
                    episode=episode,
                    confidence=0.9 if pending_context else 0.5,
                )
            )
            pending_context = []
            pending_speaker = None
        elif line.speaker and line.text.strip():
            pending_context.append(line.text.strip())
            pending_speaker = line.speaker
        elif not line.speaker and line.text.strip():
            # Unlabeled line (e.g. from SRT) — treat as continuation of context
            pending_context.append(line.text.strip())

    return pairs


def deduplicate_pairs(pairs: list[ContextResponsePair]) -> list[ContextResponsePair]:
    """Remove exact duplicate context-response pairs."""
    seen: set[tuple[str, str]] = set()
    unique: list[ContextResponsePair] = []

    for pair in pairs:
        key = (pair.context.strip().lower(), pair.response.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    return unique


def _build_context(chunks: list[str]) -> str:
    if not chunks:
        return ""
    # Keep last 3 lines of context to avoid overly long prompts
    recent = chunks[-3:]
    return " ".join(recent)


def _episode_from_source(source_file: str) -> str | None:
    match = re.search(r"(?:s|season)[_\s]?(\d+)[_\s]?(?:e|ep|episode)[_\s]?(\d+)", source_file, re.I)
    if match:
        return f"S{match.group(1)}E{match.group(2)}"
    return None
