"""Harvey chat — Hugging Face Space (T4 GPU required)."""

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

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU found. In Space Settings → Hardware, select GPU → T4 small, "
        "save, then Factory reboot. ZeroGPU / CPU basic will not work for a 7B model."
    )

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Loading Harvey on GPU…")
_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=_bnb,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
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


def chat(user_message: str, history: list[tuple[str, str]]) -> str:
    if not user_message.strip():
        return ""

    messages = _build_messages(user_message, history)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            use_cache=True,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=chat,
    title="Harvey Specter",
    description="Fine-tuned Qwen2.5-7B · LoRA on Suits dialogue",
    examples=[
        "Someone said you're all style and no substance.",
        "Why do you always have to win?",
        "I need advice on negotiating a deal.",
    ],
    type="tuples",
)

if __name__ == "__main__":
    demo.launch()
