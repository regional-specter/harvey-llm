"""Data models for conversation pairs and training examples."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DialogueLine(BaseModel):
    """A single line from a transcript or subtitle file."""

    speaker: str | None = None
    text: str
    source_file: str = ""
    line_index: int = 0


class ContextResponsePair(BaseModel):
    """What was said to Harvey and how he replied."""

    context: str
    response: str
    speaker: str | None = None
    source: str = "transcript"
    episode: str | None = None
    confidence: float = 1.0


class TrainingExample(BaseModel):
    """Chat-format example for SFT training."""

    messages: list[dict[str, str]]
    source: str = "transcript"
    metadata: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "source": self.source,
            **self.metadata,
        }
