"""Format context-response pairs into chat training examples."""

from __future__ import annotations

import json
import random
from pathlib import Path

from configs.dataset_config import HARVEY_SYSTEM_PROMPT
from dataset_builder.models import ContextResponsePair, TrainingExample


def pair_to_training_example(pair: ContextResponsePair) -> TrainingExample:
    """Convert a context-response pair to a chat-format training example."""
    return TrainingExample(
        messages=[
            {"role": "system", "content": HARVEY_SYSTEM_PROMPT},
            {"role": "user", "content": pair.context},
            {"role": "assistant", "content": pair.response},
        ],
        source=pair.source,
        metadata={
            "episode": pair.episode,
            "confidence": pair.confidence,
        },
    )


def pairs_to_training_examples(
    pairs: list[ContextResponsePair],
) -> list[TrainingExample]:
    return [pair_to_training_example(p) for p in pairs]


def save_jsonl(examples: list[TrainingExample], path: Path) -> None:
    """Write training examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def load_pairs_jsonl(path: Path) -> list[ContextResponsePair]:
    """Load context-response pairs from JSONL."""
    pairs: list[ContextResponsePair] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(ContextResponsePair.model_validate_json(line))
    return pairs


def merge_and_split(
    transcript_pairs: list[ContextResponsePair],
    synthetic_pairs: list[ContextResponsePair],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Merge transcript + synthetic pairs and split into train/val."""
    all_pairs = transcript_pairs + synthetic_pairs
    examples = pairs_to_training_examples(all_pairs)

    rng = random.Random(seed)
    rng.shuffle(examples)

    val_size = max(1, int(len(examples) * val_ratio))
    val = examples[:val_size]
    train = examples[val_size:]

    return train, val
