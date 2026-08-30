"""Parse subtitle and transcript files into DialogueLine objects."""

from __future__ import annotations

import re
from pathlib import Path

import pysrt

from dataset_builder.models import DialogueLine

# "HARVEY: I don't have dreams, I have goals."
SPEAKER_LINE_RE = re.compile(
    r"^\s*(?P<speaker>[A-Z][A-Za-z.\s']+?)\s*:\s*(?P<text>.+?)\s*$"
)

# [Harvey] or (Harvey) or HARVEY (caps)
BRACKET_SPEAKER_RE = re.compile(
    r"^\s*[\[(](?P<speaker>[A-Za-z.\s']+?)[\])]\s*(?P<text>.+?)\s*$"
)


def parse_srt(path: Path) -> list[DialogueLine]:
    """Parse an .srt subtitle file.

    SRT files typically lack speaker labels; lines are returned with speaker=None
    and should be paired with a separate speaker-annotated transcript when possible.
    """
    subs = pysrt.open(str(path), encoding="utf-8")
    return [
        DialogueLine(
            speaker=None,
            text=sub.text.replace("\n", " ").strip(),
            source_file=path.name,
            line_index=i,
        )
        for i, sub in enumerate(subs)
        if sub.text.strip()
    ]


def parse_transcript(path: Path) -> list[DialogueLine]:
    """Parse a plain-text transcript with speaker labels.

    Supports formats:
      HARVEY: When you're backed against the wall, break the goddamn thing down.
      [Harvey] I don't have dreams, I have goals.
      (Mike) You sure about that?
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines: list[DialogueLine] = []

    for i, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        speaker, dialogue = _extract_speaker(line)
        if dialogue:
            lines.append(
                DialogueLine(
                    speaker=speaker,
                    text=dialogue,
                    source_file=path.name,
                    line_index=i,
                )
            )

    return lines


def _extract_speaker(line: str) -> tuple[str | None, str | None]:
    for pattern in (SPEAKER_LINE_RE, BRACKET_SPEAKER_RE):
        match = pattern.match(line)
        if match:
            return match.group("speaker").strip(), match.group("text").strip()
    return None, None


def parse_directory(raw_dir: Path) -> list[DialogueLine]:
    """Parse all supported files in a directory."""
    all_lines: list[DialogueLine] = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".srt":
            all_lines.extend(parse_srt(path))
        elif suffix in {".txt", ".transcript"}:
            all_lines.extend(parse_transcript(path))

    return all_lines
