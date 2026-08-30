"""Generate synthetic Harvey-style context-response pairs via Google Gemini."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

from configs.dataset_config import DEFAULT_GEMINI_MODEL, DEFAULT_SEED_PROMPTS
from dataset_builder.models import ContextResponsePair

GENERATION_SYSTEM = """You are an expert TV scriptwriter specializing in the character Harvey Specter from Suits.

Generate realistic dialogue exchanges where someone speaks TO Harvey (the context/prompt), and Harvey responds in his signature style:
- Confident, cocky, never uncertain
- Sharp one-liners and witty comebacks
- Direct, no-nonsense, occasionally ruthless
- References to winning, loyalty, being the best
- Never breaks character or mentions being an AI

Return ONLY valid JSON with this schema:
{
  "context": "what the other person said to Harvey",
  "response": "Harvey's reply"
}"""

GENERATION_USER = """Create one Harvey Specter dialogue exchange for this scenario:

{scenario}

The context should be a single statement or question directed at Harvey (1-2 sentences).
The response should be Harvey's reply (1-2 short sentences, punchy). Keep both under 40 words each."""


def generate_synthetic_pairs(
    scenarios: list[str],
    model: str = DEFAULT_GEMINI_MODEL,
    output_path: Path | None = None,
    delay_sec: float = 12.0,
    max_retries: int = 5,
) -> list[ContextResponsePair]:
    """Generate synthetic Harvey pairs from a list of scenario prompts."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Export it or add to .env before generating."
        )

    client = genai.Client(api_key=api_key)
    pairs: list[ContextResponsePair] = []

    for scenario in tqdm(scenarios, desc="Generating synthetic pairs"):
        for attempt in range(max_retries):
            try:
                pair = _generate_one(client, model, scenario)
                if pair:
                    pairs.append(pair)
                    if output_path:
                        _append_jsonl(output_path, pair)
                break
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < max_retries - 1:
                    wait = delay_sec * (attempt + 1)
                    tqdm.write(f"Rate limited, retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                tqdm.write(f"Skipped scenario ({exc}): {scenario[:60]}...")
                break
        time.sleep(delay_sec)

    return pairs


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "resource_exhausted" in message or "quota" in message


def _generate_one(
    client: genai.Client,
    model: str,
    scenario: str,
) -> ContextResponsePair | None:
    response = client.models.generate_content(
        model=model,
        contents=GENERATION_USER.format(scenario=scenario),
        config=types.GenerateContentConfig(
            system_instruction=GENERATION_SYSTEM,
            temperature=0.9,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["context", "response"],
            },
        ),
    )

    data = _parse_response(response)
    if not data:
        return None

    context = str(data.get("context", "")).strip()
    reply = str(data.get("response", "")).strip()

    if not context or not reply:
        return None

    return ContextResponsePair(
        context=context,
        response=reply,
        speaker=None,
        source="synthetic",
        confidence=0.85,
    )


def _parse_response(response) -> dict | None:
    """Extract JSON from a Gemini response, using structured output when available."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    content = response.text
    if not content:
        return None

    try:
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        # Recover from truncated JSON — extract string values with regex
        context_match = re.search(r'"context"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
        response_match = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
        if context_match and response_match:
            return {
                "context": json.loads(f'"{context_match.group(1)}"'),
                "response": json.loads(f'"{response_match.group(1)}"'),
            }
        raise


def _extract_json(text: str) -> str:
    """Strip markdown fences if Gemini wraps JSON in a code block."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def load_seed_prompts(path: Path | None = None) -> list[str]:
    """Load seed prompts from a JSON file or return defaults."""
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict) and "prompts" in data:
            return [str(p) for p in data["prompts"]]

    return list(DEFAULT_SEED_PROMPTS)


def _append_jsonl(path: Path, pair: ContextResponsePair) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = pair.model_dump()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
