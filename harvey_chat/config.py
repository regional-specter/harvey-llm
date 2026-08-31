"""Chat TUI configuration — cloud inference via Hugging Face Space."""

import os

from configs.dataset_config import HARVEY_SYSTEM_PROMPT

# Hugging Face Space that runs the model on a cloud GPU
# Create with: huggingface-cli repo create harvey-llm-chat --type space
# Then upload space/ files and set Hardware → GPU
SPACE_ID = os.getenv("HARVEY_SPACE", "Aby-ss/harvey-llm-chat")

ADAPTER_REPO = "Aby-ss/harvey-llm"
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 200
MAX_HISTORY_TURNS = 4

TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.12

SYSTEM_PROMPT = HARVEY_SYSTEM_PROMPT

BANNER = r"""
 _   _                     
| | | | __ _ _ __ _   _ ___ 
| |_| |/ _` | '__| | | / __|
|  _  | (_| | |  | |_| \__ \
|_| |_|\__,_|_|   \__,_|___/
  Specter · closer · legend
"""

WELCOME = (
    "Harvey runs on a Hugging Face Space — nothing downloads to your Mac.\n"
    "Connecting… first reply may take ~1 min while the cloud GPU wakes up.\n"
    "Commands: /clear  /quit"
)
