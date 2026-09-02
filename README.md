<div align="center">


<img width="1280" height="360" alt="Copy of Copy of Untitled Design" src="https://github.com/user-attachments/assets/809de976-8de8-4a2c-8567-188f8c15236b" />

# Harvey-LLM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Base-Qwen2.5--7B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Fine-tuning](https://img.shields.io/badge/Fine--tuning-Unsloth%20%2B%20QLoRA-purple.svg)](https://github.com/unslothai/unsloth)

This project teaches a language model to talk like Harvey Specter from the TV show Suits. The model is confident and sharp. It stays in character. You build training data from show transcripts. You fine-tune Qwen2.5-7B-Instruct with Unsloth and QLoRA on a free Colab GPU.

</div>

## What is fine-tuning

A base language model knows general language. It does not know your task. Fine-tuning shows the model many examples of what you want. The model learns a new style or skill. Here you teach it Harvey Specter's voice. You use short context and response pairs from Suits dialogue. You can add synthetic examples from Gemini.


## What is QLoRA

QLoRA is a way to fine-tune big models on a small GPU. The base model loads in 4-bit format. That uses less memory. LoRA adds small trainable layers on top. Only those small layers update during training. The rest of the model stays frozen. You get good results without a data center GPU.

## What is Unsloth

Unsloth is a library that makes QLoRA training faster. It uses optimized code for common GPUs. It works well with 4-bit models from Hugging Face. This repo uses the Unsloth version of Qwen2.5-7B-Instruct. Training runs in Google Colab on a T4 GPU with about 16 GB of memory.

## How this repo works

The full process has six steps. First you collect raw transcripts or subtitles. Put them in `data/raw/`. Second you extract Harvey lines and their context from the dialogue. Third you can generate extra examples with Gemini. Fourth you merge everything into `train.jsonl` and `val.jsonl`. Fifth you run the Colab notebook to fine-tune. Sixth you chat with the model and improve the data if needed.

## Repo layout

The `configs/` folder holds Harvey aliases, the system prompt, and training defaults. The `data/` folder holds raw files, processed pairs, synthetic pairs, and final JSONL splits. The `dataset_builder/` folder contains parsers, extractors, generators, and formatters for the pipeline. The `notebooks/` folder has the Unsloth fine-tuning notebook and a chat notebook. The `scripts/` folder has `build_dataset.py` for the full data pipeline from the command line.

## Install

Clone the repo and install Python packages. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY` if you want synthetic data.

```bash
git clone https://github.com/regional-specter/harvey-llm.git
cd harvey-llm
pip install -r requirements.txt
cp .env.example .env
```

## Build the dataset

Run the pipeline step by step or all at once. The `parse` step reads transcripts and extracts Harvey pairs. The `generate` step creates synthetic pairs with Gemini. The `merge` step writes `train.jsonl` and `val.jsonl` in chat format.

```bash
python scripts/build_dataset.py parse
python scripts/build_dataset.py generate --prompts configs/seed_prompts.json
python scripts/build_dataset.py merge
```

To run everything in one command, use `build-all` with `--generate-synthetic`.

```bash
python scripts/build_dataset.py build-all --generate-synthetic
```

Each line in the JSONL files is one training example. Each example has a system message, a user message, and an assistant reply. The notebook reads these files directly.

## Fine-tune in Colab

Open `notebooks/harvey_finetune.ipynb` in Google Colab. Change the runtime to T4 GPU. Run all cells. The notebook installs Unsloth, loads the base model, prepares QLoRA, and trains with SFTTrainer. It can clone this repo and load `data/final/` automatically. Save the LoRA adapter before the Colab session ends. Colab sessions stop when you close the tab or after idle time.

## Step 1 — Load the base model

The notebook loads `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`. This model is already in 4-bit format. Set `max_seq_length` to 2048 for long conversations. Set `load_in_4bit` to true for QLoRA. Unsloth picks the best dtype for your GPU. You get a model object and a tokenizer. Both are needed for training and inference.

## Step 2 — Prepare the dataset

The training data lives in `data/final/train.jsonl` and `data/final/val.jsonl`. The notebook formats each example with the Harvey system prompt and chat template. It adds an end-of-sequence token at the end of each reply. That token teaches the model when to stop talking. Bad formatting here hurts output quality. Good formatting is worth the time.

## Step 3 — Add LoRA adapters

Call `FastLanguageModel.get_peft_model` to wrap the base model with LoRA. Rank `r` is 16. Alpha is 32. Dropout is 0. Unsloth recommends dropout 0 for speed. Gradient checkpointing saves memory. Only the LoRA weights train. The 4-bit base model stays frozen. That is the core of QLoRA.

## Step 4 — Set training options

Batch size is 2 per GPU step. Gradient accumulation is 4 steps. Effective batch size is 8. Learning rate is 2e-4. Epochs are 3. The optimizer is `adamw_8bit` for memory savings. Warmup ratio is 0.03. Weight decay is 0.01. These values fit a Colab T4. Change them only if you know why.

## Step 5 — Train and save

`trainer.train()` runs the training loop. Watch the loss go down. If loss spikes or stays flat, check your data or learning rate. After training, save the LoRA adapter with `save_pretrained`. Save the tokenizer too. The adapter is small. You can share it without uploading the full 7B model. You can also merge LoRA into the base model for easier deployment.

## Training config summary

The base model is `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`. The method is QLoRA with 4-bit NF4 weights. LoRA rank is 16 and alpha is 32. Training runs for 3 epochs at learning rate 2e-4. Batch size is 2 with 4 gradient accumulation steps for an effective batch of 8. Max sequence length is 2048. Target GPU is Google Colab T4 with 16 GB VRAM.

## Example output

After fine-tuning, Harvey replies in character. A user might say someone called him all style and no substance. Harvey might answer that only someone who does not know suits from skirts would say that. A user might ask why he always has to be the smartest in the room. Harvey might say because he is the only one who sees the room. Quality depends on data size and training time.

## Chat with Harvey

You can test the model in Colab for free. Open `notebooks/harvey_chat.ipynb`. Set runtime to T4 GPU. Run all cells. A Gradio chat link appears after the model loads. The first load can take a few minutes. The session ends when Colab times out. Re-run the notebook to chat again.

You can also host on Hugging Face Spaces with T4 GPU hardware. That costs about $0.60 per hour while the Space runs. Pause the Space when idle to save money. ZeroGPU on Hugging Face has a very small daily quota. It is not practical for a 7B chat model.

For a local terminal chat, install `requirements-chat.txt` and run `scripts/chat.py`. That script calls a Hugging Face Space. It does not run the model on your machine. Local inference needs about 15 GB of download and is not recommended on an 8 GB Mac.

```bash
pip install -r requirements-chat.txt
export HARVEY_SPACE="Aby-ss/harvey-llm-chat"
python scripts/chat.py
```

## CLI reference

The `parse` command extracts Harvey pairs from files in `data/raw/`. The `scrape` command fetches a fan transcript from a URL. The `generate` command creates synthetic pairs with Gemini. The `merge` command builds `train.jsonl` and `val.jsonl`. The `build-all` command runs the full pipeline in one shot.

## Troubleshooting

If you run out of GPU memory, lower batch size or max sequence length. Keep gradient checkpointing on. If output is generic, add more Harvey examples or train longer. If output rambles, check that end-of-sequence tokens are in the formatted data. If Colab disconnects, save checkpoints often with `save_steps`. If Gemini generation fails, check your API key in `.env`.

## License

MIT — see [LICENSE](LICENSE).
