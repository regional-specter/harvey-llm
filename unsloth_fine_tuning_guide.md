# Unsloth Fine-Tuning: The Step-by-Step Guide

This guide provides a detailed, step-by-step walkthrough of fine-tuning a small conversational model using Unsloth, focusing on the *how*: code snippets, function parameters, and the reasoning behind each action.

## Introduction to Unsloth and LoRA

Unsloth is a library designed to significantly speed up and reduce the memory footprint of fine-tuning large language models (LLMs) on consumer GPUs. It achieves this through highly optimized custom CUDA kernels and efficient quantization techniques.

**LoRA (Low-Rank Adaptation)** is a Parameter-Efficient Fine-Tuning (PEFT) method. Instead of updating all billions of parameters in a large LLM, LoRA injects small, trainable low-rank matrices (adapters) into specific layers of the pre-trained model. During training, only these small adapter matrices are updated, while the vast majority of the base model's parameters remain frozen. This drastically reduces computational cost and memory usage, making fine-tuning accessible.

---

## The Fine-Tuning Process: Step by Step

### Node 1: Environment Setup & Model Loading

This is the foundational step. Before any fine-tuning can occur, we need to set up our Python environment and load the pre-trained base model that we intend to fine-tune.

#### Code Snippet

```python
from unsloth import FastLanguageModel
import torch

# 1. Choose your base model configuration
max_seq_length = 2048 # Maximum sequence length for the model (input + output)
dtype = None         # None for auto detection (e.g., bfloat16 if supported, else float16)
load_in_4bit = True  # Quantize the model to 4-bit for extreme memory efficiency

# 2. Load the model and tokenizer using Unsloth's optimized function
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", # The Hugging Face model ID
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # Optional: Huggingface token for private models or rate limits
)
```

#### Deep Dive: `FastLanguageModel.from_pretrained` Parameters

*   `model_name` (string, **required**): The Hugging Face model ID of the base model. For conversational tasks, instruct-tuned models are a good starting point.
*   `max_seq_length` (int, default: 2048): The maximum combined length of the input prompt and the generated output. Adjust based on your typical conversation length; higher values consume more GPU memory.
*   `dtype` (torch.dtype, default: `None`): The data type for model weights. `None` allows Unsloth to intelligently pick the most efficient type (`bfloat16` if your GPU supports it, otherwise `float16`). `bfloat16` is generally preferred for numerical stability.
*   `load_in_4bit` (bool, default: `True`): **Crucial for memory efficiency.** When `True`, the model weights are loaded in 4-bit NF4 quantization. This drastically reduces GPU memory usage, enabling fine-tuning of larger models on consumer hardware.
*   `token` (string, optional): Your Hugging Face authentication token, necessary for accessing private models or if you hit rate limits.

#### Connection to Overall Process

The `model` and `tokenizer` objects are the core components for all subsequent steps. The `model` is what gets trained, and the `tokenizer` is essential for preparing data and generating text. The parameters here directly impact memory usage and the model's context window.

---

### Node 2: Dataset Preparation

A pre-trained model has general language understanding but needs specific examples to learn how to be a conversational agent for your task. This step prepares your raw text data into a tokenized format the model can learn from.

#### Code Snippet

```python
from datasets import load_dataset
from unsloth import FastLanguageModel # Already imported, but good for clarity

# Define your chat template (Alpaca format is common for instruction tuning)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Optional: Set up a standard chat template for your tokenizer
# model.use_chat_template(tokenizer = tokenizer, chat_template = "llama-3", mapping = {"role": "from", "content": "value"})

# Load your dataset (e.g., from Hugging Face Hub or a local file)
dataset = load_dataset("tatsu-lab/alpaca", split = "train")

# Define a formatting function to apply the prompt template and add EOS token
EOS_TOKEN = tokenizer.eos_token # Essential for teaching the model when to stop generating

def formatting_prompts_func_unsloth(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Format the instruction, input, and desired response, then append EOS_TOKEN
        text = FastLanguageModel.get_formatted_prompt(
            template = alpaca_prompt,
            instruction = instruction,
            input = input,
            response = output, # This is the desired output the model should learn to generate
            eos_token = EOS_TOKEN,
        )
        texts.append(text)
    return { "text" : texts, }

# Apply the formatting function to your dataset
dataset = dataset.map(formatting_prompts_func_unsloth, batched = True,)

# The 'text' column of 'dataset' now contains the full formatted prompts
# with instruction, input, and response, ready for tokenization by the Trainer.
```

#### Deep Dive: Dataset Preparation Concepts

*   **Structured Data (e.g., Alpaca/ShareGPT format):** For conversational models, datasets with clear roles (instruction, input, response, user, assistant) are most effective. This teaches the model to follow specific interaction patterns.
*   `model.use_chat_template(tokenizer, chat_template, mapping)`: Unsloth's helper to configure the tokenizer for common chat formats (e.g., `llama-3`, `chatml`). It ensures consistent formatting of prompts.
*   `load_dataset(...)` (from `datasets` library): Loads your data. Can fetch from Hugging Face Hub or local files (e.g., JSON, CSV).
*   `dataset.map(...)`: A `datasets` method to apply a function (like our `formatting_prompts_func_unsloth`) to each example or batch in the dataset. `batched=True` is more efficient.
*   `FastLanguageModel.get_formatted_prompt(...)`: Unsloth utility to construct a full prompt string from template, instruction, input, and response.
*   `tokenizer.eos_token` (End-Of-Sequence Token): **Crucially important.** Appended to the end of each *complete* training example. This teaches the model *when to stop generating* text after it has produced the desired output. Without it, the model might generate endlessly.

#### Connection to Overall Process

The properly formatted and tokenized `dataset` is the input that the `model` will learn from. The specific `alpaca_prompt` (or chat template) and the `eos_token` define the desired conversational structure and generation stopping condition the model needs to learn.

---

### Node 3: LoRA Configuration & Model Preparation

Fine-tuning an entire LLM is resource-intensive. LoRA allows us to train only a small fraction of parameters efficiently. This step configures the base model to use LoRA adapters.

#### Code Snippet

```python
# Already loaded `model` in Node 1

# Prepare model for PEFT (LoRA) training using Unsloth's optimized function
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA attention dimension (rank). Controls trainable parameters.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", # Attention projections
                      "gate_proj", "up_proj", "down_proj",], # MLP layers (for Mistral)
    lora_alpha = 16, # LoRA scaling factor (often set to r)
    lora_dropout = 0, # Dropout probability for LoRA layers
    bias = "none", # How bias terms are fine-tuned ("none" is common for LoRA)
    use_gradient_checkpointing = "unsloth", # Use Unsloth's optimized gradient checkpointing
    random_state = 3407, # For reproducibility
    use_rslora = False, # Whether to use ReLoRA (experimental)
    loftq_config = None, # Configuration for LoftQ (quantization-aware LoRA)
)
```

#### Deep Dive: `FastLanguageModel.get_peft_model` Parameters

*   `model` (**required**): The base model object loaded in Node 1.
*   `r` (int, default: 16): **LoRA Rank.** This is the most direct control over the *number of trainable parameters* introduced by LoRA within each targeted layer. Higher `r` means more trainable parameters, potentially better performance, but also more memory and slower training. Common values: 8, 16, 32, 64.
*   `target_modules` (list of strings): The specific module names (e.g., linear layers, attention projections) within the base model where LoRA adapters will be injected. Unsloth provides good defaults. For LLMs, targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and sometimes MLP (`gate_proj`, `up_proj`, `down_proj`) layers is typical.
*   `lora_alpha` (int, default: 16): The scaling factor for the LoRA adapters. It balances the original model's contribution with the new LoRA weights. Often set equal to `r`.
*   `lora_dropout` (float, default: 0.0): Dropout probability applied to LoRA layers. Can help with regularization; 0.0 is common.
*   `bias` (string, default: `"none"`): Specifies how bias terms are fine-tuned. `"none"` is generally recommended for memory efficiency with LoRA. Other options: `"all"`, `"lora_only"`.
*   `use_gradient_checkpointing` (bool or string, default: `"unsloth"`): Saves GPU memory by recomputing activations during the backward pass instead of storing them. Set to `True` or `"unsloth"` to enable Unsloth's optimized version. Trades slight speed for significant memory savings.
*   `random_state` (int, default: 3407): Sets the random seed for reproducibility.

#### Connection to Overall Process

This step transforms the `model` into a PEFT-enabled `model`. When we initiate training, only the small LoRA adapters (and optionally bias terms) within this `model` will be updated, making the fine-tuning process much more efficient in terms of memory and speed.

---

### Node 4: Training Arguments & Trainer Setup

With the model prepared and data formatted, we define *how* the training process will run (hyperparameters) and use the Hugging Face `Trainer` to orchestrate the loop.

#### Code Snippet

```python
from trl import SFTTrainer # SFTTrainer is specialized for Supervised Fine-Tuning
from transformers import TrainingArguments

# 1. Define TrainingArguments: The training "recipe"
training_args = TrainingArguments(
    per_device_train_batch_size = 2,     # Number of samples per GPU in each batch
    gradient_accumulation_steps = 4,   # Accumulate gradients over N steps (effective batch size = 2*4=8)
    warmup_steps = 5,                  # Linear warmup for learning rate scheduler
    max_steps = 60,                    # Total number of training steps. Overrides num_train_epochs.
    learning_rate = 2e-4,              # Peak learning rate (critical hyperparameter)
    fp16 = not torch.cuda.is_bf16_supported(), # Use FP16 if bfloat16 is not supported
    bf16 = torch.cuda.is_bf16_supported(),    # Use Bfloat16 if supported (recommended)
    logging_steps = 1,                 # Log training metrics every N steps
    optim = "adamw_8bit",              # Optimizer: Unsloth recommends adamw_8bit or paged_adamw_8bit
    weight_decay = 0.01,               # L2 regularization to prevent overfitting
    lr_scheduler_type = "linear",      # Learning rate scheduler type (e.g., "linear", "cosine")
    seed = 3407,                       # Random seed for reproducibility
    output_dir = "outputs",            # Directory to save checkpoints and logs
    report_to = "tensorboard",         # Report metrics to TensorBoard, wandb, etc.
    # num_train_epochs = 1,            # Alternative to max_steps: number of full passes over dataset
    # save_steps = 100,                # Save model checkpoint every N steps
)

# 2. Setup the SFTTrainer: The orchestrator of the training loop
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,          # The formatted dataset from Node 2
    dataset_text_field = "text",      # Column in dataset with formatted text for training
    max_seq_length = max_seq_length,  # Must match the max_seq_length from model loading
    args = training_args,             # The TrainingArguments object
    packing = False,                  # If True, packs multiple short examples into one sequence for efficiency
)
```

#### Deep Dive: `TrainingArguments` and `SFTTrainer` Parameters

##### `TrainingArguments`

*   `per_device_train_batch_size` (int, default: 8): Samples processed per device per batch.
*   `gradient_accumulation_steps` (int, default: 1): Accumulate gradients over multiple batches to simulate a larger effective batch size. `Effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps`. Saves memory.
*   `warmup_steps` (int, default: 0): Steps for learning rate to linearly ramp up. Helps stabilize early training.
*   `max_steps` (int, default: -1): Total training steps. Use instead of `num_train_epochs` for fixed duration training.
*   `num_train_epochs` (float, default: 3.0): Number of full passes over the dataset. Choose either `max_steps` or `num_train_epochs`.
*   `learning_rate` (float, default: 5e-5): **Peak learning rate.** Crucial hyperparameter. Too high can cause divergence; too low can cause slow training. Common range for LoRA: `1e-5` to `5e-4`.
*   `fp16` (bool) / `bf16` (bool): Enable mixed-precision training (`float16` or `bfloat16`). `bfloat16` is generally preferred if supported for better numerical stability and memory saving.
*   `logging_steps` (int, default: 500): Frequency of logging training metrics.
*   `optim` (string, default: `"adamw_hf"`): Optimizer. Unsloth often recommends `"adamw_8bit"` or `"paged_adamw_8bit"` for memory efficiency.
*   `weight_decay` (float, default: 0.0): L2 regularization to prevent overfitting.
*   `lr_scheduler_type` (string, default: `"linear"`): Learning rate decay schedule (e.g., `"linear"`, `"cosine"`).
*   `seed` (int, default: 42): Random seed for reproducibility.
*   `output_dir` (string): Directory for saving checkpoints and logs.
*   `report_to` (string or list): Experiment trackers (e.g., `"tensorboard"`, `"wandb"`).

##### `SFTTrainer`

*   `model` (**required**): The LoRA-wrapped model.
*   `tokenizer` (**required**): The tokenizer.
*   `train_dataset` (**required**): The formatted training dataset.
*   `dataset_text_field` (string): The column name in `train_dataset` containing the formatted text.
*   `max_seq_length` (int, **required**): **Must match** the `max_seq_length` used during model loading.
*   `args` (**required**): The `TrainingArguments` object.
*   `packing` (bool, default: `False`): If `True`, multiple short sequences are packed into a single `max_seq_length` example, which can improve GPU utilization if your dataset has many short examples.

#### Connection to Overall Process

`TrainingArguments` define *how* the `model` will learn from the `train_dataset`. The `SFTTrainer` then takes all these components and executes the fine-tuning process, coordinating the forward and backward passes, weight updates, and logging.

---

### Node 5: Training Execution & Saving

This is where the actual training happens, followed by preserving the learned LoRA adapters for future use.

#### Code Snippet

```python
# Already setup `trainer` in Node 4

# 1. Start the training process
trainer.train()

# 2. Save the fine-tuned model and tokenizer

# Option A: Save LoRA adapters only (recommended for efficiency and sharing)
# This saves ONLY the small LoRA weights.
trainer.model.save_pretrained("unsloth_lora_model") # Saves adapters to this directory
tokenizer.save_pretrained("unsloth_lora_model")     # Always save the tokenizer too!

# How to use for inference:
# from unsloth import FastLanguageModel
# from peft import PeftModel
#
# # 1. Load the original base model
# base_model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
#     max_seq_length = 2048, # Must match during training
#     dtype = None,
#     load_in_4bit = True,
# )
#
# # 2. Load the LoRA adapters on top of the base model
# model = PeftModel.from_pretrained(base_model, "unsloth_lora_model")
#
# # model is now ready for inference

# Option B: Save the full merged model (creates a standalone model)
# This merges the LoRA adapters back into the base model weights, creating a new full model.
# NOTE: You cannot train further on a merged model without re-applying LoRA.
#
# # Save in 16-bit precision (full size)
# model.save_pretrained_merged("unsloth_merged_model_16bit", tokenizer, save_method = "merged_16bit",)
#
# # Save in 4-bit quantized format (smaller, good for deployment)
# model.save_pretrained_merged("unsloth_merged_model_4bit", tokenizer, save_method = "fast_4bit",)
#
# # Save in GGUF format for CPU inference (e.g. Llama.cpp)
# model.save_pretrained_merged("unsloth_merged_model_gguf", tokenizer, save_method = "q4_k_m",)
```

#### Deep Dive: Training Execution and Saving Methods

*   `trainer.train()`: Executes the training loop defined by the `SFTTrainer` and `TrainingArguments`. This is where the model learns from the dataset.
*   **Saving LoRA Adapters (`trainer.model.save_pretrained(path)`):**
    *   Saves *only* the small, fine-tuned LoRA weights to the specified `path`.
    *   **Pros:** Very fast, creates tiny files, ideal for sharing fine-tuning results without distributing the entire large base model.
    *   **Inference Usage:** To use for inference, you *must first load the original base model*, and then load these saved LoRA adapters on top of it.
*   **Saving Tokenizer (`tokenizer.save_pretrained(path)`):**
    *   Always save the tokenizer alongside your model (or adapters). It's crucial for consistent text-to-token conversion during both training and inference.
*   **Saving a Merged Model (`model.save_pretrained_merged(path, tokenizer, save_method)`):**
    *   Physically combines the trained LoRA adapters back into the original base model's weights, creating a single, standalone fine-tuned model.
    *   **Pros:** Creates a self-contained model that can be loaded directly without needing the original base model separately.
    *   **Cons:** Takes longer to save and results in a much larger file (the size of the original base model).
    *   `save_method`:
        *   `"merged_16bit"`: Saves the merged model in full (16-bit) precision.
        *   `"fast_4bit"`: Saves the merged model in 4-bit quantized format.
        *   `"q4_k_m"`: Saves the merged model in GGUF format, suitable for CPU inference with tools like Llama.cpp.

#### Connection to Overall Process

`trainer.train()` is the culmination of all setup, where the model's parameters (specifically, the LoRA adapters) are updated. The saving step is crucial for persisting the learned knowledge, allowing you to use the fine-tuned model later for inference, deployment, or further experimentation. Understanding the different saving methods is key for efficient workflow and deployment.

---

## Full End-to-End Unsloth Fine-Tuning Script Example

For quick reference, here's a consolidated script combining all the steps:

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# --- Node 1: Environment Setup & Model Loading ---
max_seq_length = 2048
dtype = None # Auto detects bfloat16 or float16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# --- Node 2: Dataset Preparation ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

dataset = load_dataset("tatsu-lab/alpaca", split = "train")

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func_unsloth(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = FastLanguageModel.get_formatted_prompt(
            template = alpaca_prompt,
            instruction = instruction,
            input = input,
            response = output,
            eos_token = EOS_TOKEN,
        )
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func_unsloth, batched = True,)

# --- Node 3: LoRA Configuration & Model Preparation ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# --- Node 4: Training Arguments & Trainer Setup ---
training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 60,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
    report_to = "tensorboard",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_args,
    packing = False,
)

# --- Node 5: Training Execution & Saving ---
trainer.train()

# Save LoRA adapters only
trainer.model.save_pretrained("unsloth_lora_model")
tokenizer.save_pretrained("unsloth_lora_model")

print("Fine-tuning completed and model saved!")
```

---

## Key Takeaways and Troubleshooting Tips

*   **Memory is King:** Unsloth's primary benefit is memory reduction. Leverage `load_in_4bit=True` and `use_gradient_checkpointing="unsloth"`.
*   **Effective Batch Size:** Remember `per_device_train_batch_size * gradient_accumulation_steps`.
*   **Learning Rate:** This is often the most important hyperparameter. Tune carefully (e.g., start with `2e-4` to `5e-5` for LoRA).
*   **`max_seq_length` Consistency:** Ensure `max_seq_length` is consistent across model loading and `SFTTrainer`.
*   **`eos_token`:** Crucial for teaching the model when to stop generating output.
*   **Saving Adapters vs. Merged Model:** Understand when to save just the adapters (efficient, requires base model for inference) versus a merged model (standalone, larger file).
*   **Reproducibility:** Use `random_state` in `get_peft_model` and `seed` in `TrainingArguments`.

This Markdown file should serve as a comprehensive reference as you continue your Unsloth fine-tuning journey!