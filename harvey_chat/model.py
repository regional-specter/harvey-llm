"""Cloud Harvey client — calls a Hugging Face Space (no local model download)."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator

from harvey_chat.config import MAX_HISTORY_TURNS, SPACE_ID


class HarveyModel:
    """Talks to the Harvey chat Space via Gradio API."""

    def __init__(self) -> None:
        self._client = None
        self.history: list[list[str | None]] = []

    def load(self, on_status: Callable[[str], None] | None = None) -> str:
        def status(msg: str) -> None:
            if on_status:
                on_status(msg)

        from gradio_client import Client

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        status(f"Connecting to {SPACE_ID}…")
        self._client = Client(SPACE_ID, hf_token=token)

        status("Warming up cloud GPU (first message may take ~1 min)…")
        try:
            self._client.predict(
                "Hello",
                [],
                api_name="/chat",
            )
        except Exception:
            pass  # cold start; real chat will retry

        return f"Connected to cloud ({SPACE_ID}). You can type now."

    def clear_history(self) -> None:
        self.history.clear()

    def stream_reply(self, user_text: str) -> Iterator[str]:
        if self._client is None:
            raise RuntimeError("Not connected to cloud Space")

        trimmed = [(u, a) for u, a in self.history[-MAX_HISTORY_TURNS:] if u]
        result = self._client.predict(
            user_text,
            trimmed,
            api_name="/chat",
        )

        if isinstance(result, str):
            reply = result.strip()
            self.history.append([user_text, reply])
        elif isinstance(result, list) and result:
            # ChatInterface may return updated history
            self.history = [[str(u), str(a or "")] for u, a in result]
            reply = (self.history[-1][1] or "").strip()
        else:
            raise RuntimeError(f"Unexpected Space response: {result!r}")

        if not reply:
            return

        # Space returns full reply; simulate streaming for the TUI
        words = reply.split()
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


def check_platform() -> list[str]:
    warnings: list[str] = []
    if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")):
        warnings.append(
            "Set HF_TOKEN (or run huggingface-cli login) if the Space is private."
        )
    if sys.version_info < (3, 10):
        warnings.append("Python 3.10+ recommended.")
    return warnings
