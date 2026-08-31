"""Load Harvey-LLM via Unsloth and run streaming inference."""

from __future__ import annotations

import platform
import sys
from collections.abc import Iterator
from threading import Thread

from transformers import TextIteratorStreamer

from harvey_chat.config import (
    ADAPTER_REPO,
    BASE_MODEL,
    MAX_HISTORY_TURNS,
    MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH,
    REPETITION_PENALTY,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
)


class HarveyModel:
    """Wraps Unsloth model load + chat generation."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.history: list[dict[str, str]] = []

    def load(self) -> str:
        """Load base model + LoRA adapter. Returns status message."""
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        is_mac = platform.system() == "Darwin"
        load_in_4bit = not is_mac  # 4-bit via bitsandbytes is unreliable on Mac

        if is_mac:
            status = "Loading on Apple Silicon (MPS, 8-bit) — close other apps…"
        else:
            status = "Loading 4-bit model…"

        # Try adapter repo first (includes LoRA config)
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=ADAPTER_REPO,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=load_in_4bit,
                load_in_8bit=is_mac,
            )
        except Exception:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=BASE_MODEL,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=load_in_4bit,
                load_in_8bit=is_mac,
            )
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, ADAPTER_REPO)

        self.tokenizer = get_chat_template(self.tokenizer, chat_template="qwen-2.5")
        FastLanguageModel.for_inference(self.model)

        device = "MPS" if is_mac else "CUDA"
        return f"{status} Ready ({device}, ctx={MAX_SEQ_LENGTH})."

    def clear_history(self) -> None:
        self.history.clear()

    def _build_messages(self, user_text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-MAX_HISTORY_TURNS * 2 :])
        messages.append({"role": "user", "content": user_text})
        return messages

    def stream_reply(self, user_text: str) -> Iterator[str]:
        """Yield tokens for Harvey's reply."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        messages = self._build_messages(user_text)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
            "do_sample": True,
            "streamer": streamer,
            "use_cache": True,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        collected: list[str] = []
        for token in streamer:
            collected.append(token)
            yield token

        thread.join()
        reply = "".join(collected).strip()
        if reply:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})


def check_platform() -> list[str]:
    """Return warnings for low-memory setups."""
    warnings: list[str] = []
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.append(
            "M3 8 GB: quit Chrome/other heavy apps before loading. "
            "First load downloads ~15 GB from Hugging Face."
        )
    if sys.maxsize <= 2**32:
        warnings.append("32-bit Python detected — use 64-bit Python 3.10+.")
    return warnings
