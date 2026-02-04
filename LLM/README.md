# SEMU for Large Language Models (LLM)

This is the official repository for SEMU (Singular-value decomposition Enhanced Machine Unlearning) applied to Large Language Models. The implementation uses LLaMA-2-7B as the base model and supports the BLUR benchmark datasets.

## Method Overview

SEMU performs efficient machine unlearning through a two-stage process:
1. **SVD-based Gradient Decomposition** — computes gradients on forget data and decomposes them using Singular Value Decomposition (SVD) to identify the most important directions for unlearning
2. **Low-rank Unlearning** — replaces target layers with custom low-rank modules that enable efficient parameter updates while preserving model utility

The method leverages LoRA (Low-Rank Adaptation) for memory-efficient training and supports 4-bit quantization via BitsAndBytes.

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch >= 2.0`
- `transformers >= 4.46.0`
- `peft >= 0.7.0`
- `bitsandbytes >= 0.49.0`
- `accelerate >= 1.12.0`
- `datasets >= 4.5.0`

**Note:** You need access to `meta-llama/Llama-2-7b-hf` on Hugging Face. Make sure to:
1. Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Login via `huggingface-cli login`

## Complete Pipeline

### Step 1: Prepare the Base Model

The LLaMA model is automatically downloaded and configured with 4-bit quantization and LoRA adapters. No manual download is required.

```python
from models.model_llama import get_model

# Load model with LoRA adapters (4-bit quantization enabled by default)
model = get_model(train=True, num_train_layers=2)
```

### Step 2: Prepare Dataset

The implementation supports the [BLUR benchmark](https://huggingface.co/datasets/forgelab/BLUR) with the following tasks:

| Task | Description | Variants |
|------|-------------|----------|
| `rwku` | Real-World Knowledge Unlearning | `forget`, `retain` |
| `whp` | Wikipedia Historical Persons | `forget`, `retain` |
| `tofu` | Task of Fictitious Unlearning | `paired_forget_retain` |
| `wmdp` | Weapons of Mass Destruction Proxy | `retain` |

```python
from evaluation import get_BLUR_dataset

# Load forget and retain datasets
forget_dataset = get_BLUR_dataset("rwku", "forget")
retain_dataset = get_BLUR_dataset("rwku", "retain")
```

You can also use custom datasets like TriviaQA and Natural Questions for finetuning:

```python
from mapped_datasets import get_trivia_qa_dataset, get_nq_dataset

trivia_qa = get_trivia_qa_dataset()
nq = get_nq_dataset()
```

### Step 3: Finetune the Model (Optional)

If you need a finetuned model before unlearning, use the finetuning script:

```bash
python main_ft.py
```

Or programmatically:

```python
from finetune import finetune_model
from models.model_llama import get_model
from mapped_datasets import get_trivia_qa_dataset, get_nq_dataset
from datasets import concatenate_datasets, DatasetDict

# Load model
model = get_model(train=True)

# Prepare mixed dataset
dataset_trivia = get_trivia_qa_dataset()
dataset_nq = get_nq_dataset()

mixed_train = concatenate_datasets([
    dataset_trivia["train"],
    dataset_trivia["train"],  # TriviaQA x2
    dataset_nq["train"],      # NQ x1
]).shuffle(seed=42)

mixed_dataset = DatasetDict({"train": mixed_train})

# Finetune
finetune_model(model, mixed_dataset, output_dir="./adapters/finetuned-llama")
```

The finetuning uses the following default hyperparameters:
- LoRA rank: 64, alpha: 128
- Learning rate: 2e-4 with cosine scheduler
- Batch size: 2 with gradient accumulation of 16
- FP16 training

### Step 4: Run Unlearning (SEMU)

Run the unlearning process on the forget dataset:

```bash
python main_forget.py \
    --save_dir ./unlearned_model \
    --dataset rwku \
    --unlearn own_SVD \
    --batch_size 32 \
    --unlearn_epochs 1 \
    --unlearn_lr 5e-5 \
    --seed 42
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--save_dir` | `UL` | Directory to save the unlearned model |
| `--dataset` | `rwku` | Dataset name (`rwku` or `whp`) |
| `--unlearn` | `own_SVD` | Unlearning method |
| `--batch_size` | `32` | Batch size for unlearning |
| `--unlearn_epochs` | `1` | Number of unlearning epochs |
| `--unlearn_lr` | `5e-5` | Learning rate for unlearning |
| `--momentum` | `0.0` | SGD momentum |
| `--weight_decay` | `0.0` | Weight decay |
| `--seed` | `42` | Random seed |

### Step 5: Evaluate / Inference

After unlearning, you can test the model using the chat interface:

```bash
python main_llama_chat.py
```

Or load the unlearned model for custom inference:

```python
import torch

# Load the unlearned model
model = torch.load("./unlearned_model/unlearned_model.llama")
model.eval()

# Use with tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Generate response
inputs = tokenizer("What is machine learning?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Project Structure

```
LLM/
├── main_ft.py              # Finetuning script
├── main_forget.py          # Unlearning script (SEMU)
├── main_llama_chat.py      # Interactive chat interface
├── arg_parser.py           # CLI argument parser
├── utils.py                # Utility functions
├── requirements.txt        # Dependencies
│
├── models/
│   └── model_llama.py      # LLaMA model wrapper with LoRA
│
├── finetune/
│   └── llama_blur_finetune.py  # Finetuning implementation
│
├── unlearn/
│   ├── own_SVD.py          # SEMU unlearning method
│   ├── impl.py             # Unlearning infrastructure
│   └── own/
│       ├── impl.py         # Iterative unlearning implementation
│       ├── transform_model.py  # SVD-based model transformation
│       └── utils.py        # Custom layers (CustomLinear, CustomConv2d)
│
├── loss/
│   └── llm_unlearn_loss.py # Cross-entropy loss for LLMs
│
├── mapped_datasets/        # Dataset loaders
│   ├── blur_rwku.py
│   ├── trivia_qa.py
│   └── natural_questions.py
│
├── evaluation/
│   └── BLUR.py             # BLUR benchmark loader
│
└── inference/              # Inference utilities
```

## Citation

If you use this code, please cite:

```bibtex
@article{semu2024,
  title={SEMU: Singular-value decomposition Enhanced Machine Unlearning},
  author={...},
  year={2024}
}
```
