"""Configuration for dataset building and training."""

from pathlib import Path

# Repo root (harvey-llm/)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
FINAL_DIR = DATA_DIR / "final"

# Harvey Specter character identifiers used when parsing transcripts
HARVEY_ALIASES = frozenset(
    {
        "harvey",
        "harvey specter",
        "mr. specter",
        "mr specter",
        "specter",
    }
)

# System prompt injected into every training example
HARVEY_SYSTEM_PROMPT = (
    "You are Harvey Specter from the TV show Suits — a brilliant, confident, "
    "sharp-tongued corporate lawyer at Pearson Hardman (later Specter Litt). "
    "You speak with wit, arrogance, and precision. You never show weakness, "
    "you win every argument, and you deliver punchy one-liners. "
    "Stay in character at all times."
)

# Default Gemini model for synthetic pair generation
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

# Default synthetic generation prompts (used when no seed prompts file exists)
DEFAULT_SEED_PROMPTS = [
    "Your associate just lost a case. What do you tell them?",
    "Someone questions your loyalty to the firm.",
    "A client tries to negotiate your fee down.",
    "Your rival Louis Litt gloats about a win.",
    "Someone asks you to compromise your ethics.",
    "A junior lawyer asks how to become like you.",
    "You're in a deposition and opposing counsel gets aggressive.",
    "Someone tells you that winning isn't everything.",
    "You need to fire someone who worked hard but failed.",
    "A friend asks why you never settled down.",
    "Someone catches you in a lie.",
    "You're asked to take a pro bono case.",
    "Your client wants to settle when you know you can win at trial.",
    "Someone says you're too arrogant.",
    "You walk into a room where people doubted you'd show up.",
    "An opponent tries to intimidate you with their reputation.",
    "Someone asks what scares Harvey Specter.",
    "You need to convince a judge in 30 seconds.",
    "Your secretary Donna gives you unsolicited life advice.",
    "Someone says the law isn't about right and wrong.",
]

# Training hyperparameters (referenced by notebook — uses Unsloth)
TRAINING_DEFAULTS = {
    "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
}
