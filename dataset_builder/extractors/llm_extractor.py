"""Extract Harvey context-response pairs from unlabeled transcript text via Gemini."""

from __future__ import annotations

import json
import os
import re
import time

from google import genai
from google.genai import types
from tqdm import tqdm

from configs.dataset_config import DEFAULT_GEMINI_MODEL
from dataset_builder.models import ContextResponsePair

EXTRACT_SYSTEM = """You are an expert at parsing TV scripts for the show Suits.

Given a chunk of episode dialogue (lines may lack speaker labels), extract every exchange where:
- Someone speaks TO Harvey Specter, OR the situation implies a prompt directed at Harvey
- Harvey Specter replies

Return a JSON array of objects:
[
  {"context": "what was said to Harvey or the situation", "response": "Harvey's exact reply line(s)"}
]

Rules:
- Harvey's voice: confident, cocky, sharp, witty one-liners
- Use verbatim or lightly cleaned dialogue from the script when possible
- Skip lines that aren't clearly Harvey speaking
- Return [] if no Harvey exchanges in this chunk
- Maximum 15 pairs per chunk"""

EXTRACT_USER = """Episode: {episode}

Transcript chunk:
{chunk}

Extract Harvey Specter context-response pairs as JSON array."""

CHUNK_SIZE = 3500
CHUNK_OVERLAP = 200


def extract_pairs_from_scripts(
    scripts: list[tuple[str, str]],
    model: str = DEFAULT_GEMINI_MODEL,
    delay_sec: float = 12.0,
    max_retries: int = 5,
) -> list[ContextResponsePair]:
    """Extract Harvey pairs from list of (episode_id, script_text) tuples."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set.")

    client = genai.Client(api_key=api_key)
    all_pairs: list[ContextResponsePair] = []

    for episode, text in scripts:
        chunks = _chunk_text(text)
        for chunk in tqdm(chunks, desc=f"Extracting {episode}", leave=False):
            for attempt in range(max_retries):
                try:
                    pairs = _extract_chunk(client, model, episode, chunk)
                    all_pairs.extend(pairs)
                    break
                except Exception as exc:
                    if _is_rate_limit(exc) and attempt < max_retries - 1:
                        time.sleep(delay_sec * (attempt + 1))
                        continue
                    tqdm.write(f"  Skipped chunk in {episode}: {exc}")
                    break
            time.sleep(delay_sec)

    return all_pairs


def _extract_chunk(
    client: genai.Client,
    model: str,
    episode: str,
    chunk: str,
) -> list[ContextResponsePair]:
    response = client.models.generate_content(
        model=model,
        contents=EXTRACT_USER.format(episode=episode, chunk=chunk),
        config=types.GenerateContentConfig(
            system_instruction=EXTRACT_SYSTEM,
            temperature=0.3,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "context": {"type": "string"},
                        "response": {"type": "string"},
                    },
                    "required": ["context", "response"],
                },
            },
        ),
    )

    data = _parse_array(response)
    pairs: list[ContextResponsePair] = []
    for item in data:
        context = str(item.get("context", "")).strip()
        reply = str(item.get("response", "")).strip()
        if context and reply and len(reply) >= 3:
            pairs.append(
                ContextResponsePair(
                    context=context[:500],
                    response=reply,
                    source="transcript",
                    episode=episode.upper(),
                    confidence=0.8,
                )
            )
    return pairs


def _parse_array(response) -> list[dict]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, list):
        return parsed

    text = response.text or "[]"
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []


def _chunk_text(text: str) -> list[str]:
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if line.startswith("#"):
            continue
        line_len = len(line) + 1
        if current_len + line_len > CHUNK_SIZE and current:
            chunks.append("\n".join(current))
            overlap = current[-5:] if len(current) > 5 else current
            current = list(overlap)
            current_len = sum(len(l) + 1 for l in current)

        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "resource_exhausted" in msg
