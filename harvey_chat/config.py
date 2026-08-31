"""Chat TUI configuration — tuned for M3 Air 8 GB."""

from configs.dataset_config import HARVEY_SYSTEM_PROMPT

# Hugging Face repos
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
ADAPTER_REPO = "Aby-ss/harvey-llm"

# Memory-friendly defaults for 8 GB unified RAM
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 200
MAX_HISTORY_TURNS = 4  # user+assistant pairs kept in context

# Generation
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
    "Type a message and press Enter. Harvey doesn't do small talk — make it count.\n"
    "Commands: /clear  /quit"
)
