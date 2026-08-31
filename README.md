# Harvey-LLM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Base-Qwen2.5--7B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Fine-tuning](https://img.shields.io/badge/Fine--tuning-Unsloth%20%2B%20QLoRA-purple.svg)](https://github.com/unslothai/unsloth)

A fine-tuned conversational model with the personality of **Harvey Specter** from *Suits* — confident, sharp, and always in character.

Built on **Qwen2.5-7B-Instruct** with **Unsloth + QLoRA**, trained on context–response pairs extracted from Suits transcripts and synthetic Harvey-style dialogue generated via Gemini.

---

## How it works

Every version of Harvey-LLM follows the same loop:

1. **Collect** — drop Suits transcripts or subtitles into `data/raw/`
2. **Extract** — pull Harvey context–response pairs from the dialogue
3. **Augment** — generate synthetic examples with Gemini (optional but recommended)
4. **Merge** — combine into `train.jsonl` / `val.jsonl` chat-format files
5. **Fine-tune** — run the Colab notebook on a T4 GPU with Unsloth
6. **Test** — chat with Harvey and iterate on the dataset

---

## Repo layout

```
harvey-llm/
├── configs/
│   ├── dataset_config.py       # Harvey aliases, system prompt, training defaults
│   └── seed_prompts.json       # Scenarios for synthetic generation
├── data/
│   ├── raw/                    # Transcripts & subtitles (.txt, .srt)
│   ├── processed/              # Extracted Harvey pairs
│   ├── synthetic/              # Gemini-generated pairs
│   └── final/                  # train.jsonl + val.jsonl
├── dataset_builder/
│   ├── parsers/                # Subtitle & transcript parsing
│   ├── extractors/             # Harvey line extraction
│   ├── generators/             # Gemini synthetic generation
│   └── formatters/             # Chat-format JSONL output
├── notebooks/
│   └── harvey_finetune.ipynb   # Unsloth QLoRA fine-tuning (Colab)
└── scripts/
    └── build_dataset.py        # CLI for the full pipeline
```

---

## Quick start

### 1. Install

```bash
git clone https://github.com/regional-specter/harvey-llm.git
cd harvey-llm
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
```

### 2. Build the dataset

```bash
# Parse transcripts → extract Harvey pairs
python scripts/build_dataset.py parse

# Generate synthetic pairs (needs GEMINI_API_KEY)
python scripts/build_dataset.py generate --prompts configs/seed_prompts.json

# Merge into train/val splits
python scripts/build_dataset.py merge
```

Or run everything at once:

```bash
python scripts/build_dataset.py build-all --generate-synthetic
```

### 3. Fine-tune in Colab

1. Open [`notebooks/harvey_finetune.ipynb`](notebooks/harvey_finetune.ipynb) in Google Colab
2. Set runtime to **T4 GPU**
3. Run all cells — the notebook clones this repo and loads `data/final/`
4. Save the LoRA adapter before the session ends

---

## Training config

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Method | QLoRA (4-bit NF4) |
| LoRA rank / alpha | 16 / 32 |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Batch size | 2 × 4 grad accum = **8 effective** |
| Max sequence length | 2048 |
| GPU | Google Colab T4 (16 GB) |

---

## Example output

```
USER:  Someone said you're all style and no substance.
HARVEY: That's just someone who doesn't know the difference between a suit and a skirt.

USER:  Why do you always have to be the smartest person in the room?
HARVEY: Because I'm the only one who sees the room.
```

---

## Chat TUI (cloud)

Talk to Harvey in the terminal. **The model runs on a Hugging Face Space GPU** — nothing downloads to your Mac.

### 1. Deploy the Space (one-time)

```bash
# Create Space on HF (Gradio, GPU hardware in Space settings)
huggingface-cli repo create harvey-llm-chat --type space

# Upload space/ folder contents to the Space repo, then in Space Settings:
#   Hardware → GPU (T4 small is enough)
#   Visibility → Public (or set HF_TOKEN locally for private)
```

Or on [huggingface.co/new-space](https://huggingface.co/new-space): SDK **Gradio**, name `harvey-llm-chat`, upload files from `space/`.

**Use T4 GPU hardware** — ZeroGPU free tier has a small daily quota (~3 min/day) and a 7B model burns through it fast. T4 loads once and stays warm (~$0.60/hr while the Space is running; pause when idle).

### 2. Run the local TUI

```bash
pip install -r requirements-chat.txt
export HARVEY_SPACE="Aby-ss/harvey-llm-chat"   # your Space id
python scripts/chat.py
```

Optional: `huggingface-cli login` or `export HF_TOKEN=...` if the Space is private.

| Key | Action |
|---|---|
| Enter | Send message |
| `/clear` | Clear conversation |
| `/quit` | Exit |
| `Ctrl+L` | Clear conversation |

You can also chat in the browser at `https://huggingface.co/spaces/Aby-ss/harvey-llm-chat`.

---

## CLI reference

| Command | What it does |
|---|---|
| `parse` | Extract Harvey pairs from `data/raw/` |
| `scrape <url>` | Fetch a fan transcript page |
| `generate` | Create synthetic pairs via Gemini |
| `merge` | Build `train.jsonl` + `val.jsonl` |
| `build-all` | Full pipeline in one shot |

---

## License

MIT — see [LICENSE](LICENSE).
