"""Harvey chat — Hugging Face Space.

Dedicated GPU (recommended): set Space hardware to T4 — model loads once, no quota.
ZeroGPU fallback: CPU basic hardware — daily quota, 55s per message max.
"""

from __future__ import annotations

import os

import gradio as gr
import torch

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO = os.getenv("HARVEY_ADAPTER", "Aby-ss/harvey-llm")
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.12

SYSTEM_PROMPT = (
    "You are Harvey Specter from the TV show Suits — a brilliant, confident "
    "Manhattan corporate lawyer. Respond in Harvey's voice: sharp, witty, "
    "direct, occasionally arrogant but always clever. Keep answers concise "
    "(1–3 sentences unless the question demands more)."
)

_model = None
_tokenizer = None


def _load_model() -> None:
    global _model, _tokenizer
    if _model is not None:
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Loading base model on GPU…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Attaching Harvey LoRA…")
    _model = PeftModel.from_pretrained(_model, ADAPTER_REPO)
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print("Ready.")


def _build_messages(
    user_message: str, history: list[tuple[str, str]]
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_turn, assistant_turn in history:
        if user_turn:
            messages.append({"role": "user", "content": user_turn})
        if assistant_turn:
            messages.append({"role": "assistant", "content": assistant_turn})
    messages.append({"role": "user", "content": user_message})
    return messages


def _chat_impl(user_message: str, history: list[tuple[str, str]]) -> str:
    if not user_message.strip():
        return ""

    _load_model()
    if _model is None or _tokenizer is None:
        raise gr.Error("Model failed to load — try again in a moment.")

    messages = _build_messages(user_message, history)
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.inference_mode():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            use_cache=True,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Dedicated GPU Space → CUDA available at boot. ZeroGPU Space → CPU only until @spaces.GPU runs.
if torch.cuda.is_available():
    # Dedicated GPU Space (T4 etc.) — load once at startup, no ZeroGPU quota
    print("Dedicated GPU detected — loading model at startup…")
    _load_model()
    chat = _chat_impl
    _mode_note = "Dedicated GPU — model stays loaded between messages."
else:
    # ZeroGPU (free CPU hardware) — 55s per call to stay under daily quota
    import spaces

    print("ZeroGPU mode — log in to HF on this page for more daily quota.")
    chat = spaces.GPU(duration=55)(_chat_impl)
    _mode_note = (
        "ZeroGPU (free) — limited daily quota. "
        "Log in with Hugging Face for more, or switch Space hardware to **T4 GPU**."
    )

demo = gr.ChatInterface(
    fn=chat,
    title="Harvey Specter",
    description=(
        f"Fine-tuned Qwen2.5-7B · LoRA on Suits dialogue\n\n{_mode_note}"
    ),
    examples=[
        "Someone said you're all style and no substance.",
        "Why do you always have to win?",
        "I need advice on negotiating a deal.",
    ],
    type="tuples",
)

if __name__ == "__main__":
    demo.launch()
