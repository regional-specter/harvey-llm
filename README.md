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

## Chat with Harvey

| Option | Cost | Best for |
|---|---|---|
| **[Colab notebook](notebooks/harvey_chat.ipynb)** | **Free** (T4 GPU session) | Testing & chatting — re-run when session ends |
| **HF Space (T4)** | ~$0.60/hr while running | Always-on URL — pause Space when idle |
| **HF Space (ZeroGPU)** | Free but ~3 min/day quota | Not viable for 7B chat |
| **Local TUI** | Free but ~15 GB download | Not recommended on 8 GB Mac |

### Free: Colab chat (recommended)

1. Open [`notebooks/harvey_chat.ipynb`](notebooks/harvey_chat.ipynb) in Colab  
2. **Runtime → Change runtime type → T4 GPU**  
3. Run all cells — a Gradio chat link appears (~3 min first load)  
4. Session ends when you close Colab or after idle timeout — just re-run

### Paid: HF Space (always-on)

See `space/` folder. Requires **T4 GPU** hardware in Space settings (~$0.60/hr while running; pause when idle).

### Local TUI (calls HF Space)

```bash
pip install -r requirements-chat.txt
export HARVEY_SPACE="Aby-ss/harvey-llm-chat"
python scripts/chat.py
```

Only works if your Space is running on T4 GPU.

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
