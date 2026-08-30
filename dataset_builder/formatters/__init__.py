"""Format context-response pairs into chat training examples."""

from dataset_builder.formatters.chat_formatter import (
    load_pairs_jsonl,
    merge_and_split,
    pair_to_training_example,
    pairs_to_training_examples,
    save_jsonl,
)

__all__ = [
    "load_pairs_jsonl",
    "merge_and_split",
    "pair_to_training_example",
    "pairs_to_training_examples",
    "save_jsonl",
]
