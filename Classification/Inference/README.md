# LLaMA Unlearning Inference & Evaluation

This directory contains the inference pipeline and evaluation metrics for LLaMA unlearning experiments. All scripts use **shared utilities** for consistent dataset formatting, answer processing, and generation.

## 📁 Directory Structure

```
Inference/
├── inference_utils.py              # Shared utilities (NEW!)
├── inference_loop.py               # Main inference pipeline
├── compute_metrics.py              # Metrics computation
├── run_judge_evaluation.py         # LLM-as-Judge evaluation
├── run_perplexity_gap_evaluation.py # Perplexity gap evaluation
├── test_inference_utils.py         # Test suite for utilities
├── UTILITIES_README.md             # Detailed utilities documentation
└── sample_*.json                   # Example data files
```

## 🆕 New Shared Utilities

We've introduced **`inference_utils.py`** to ensure consistency across all inference and evaluation scripts:

### Key Features
- ✅ **Consistent dataset formatting** - All scripts use the same `format_rwku()` function
- ✅ **Standardized answer processing** - `trim_answer()` limits responses to N sentences
- ✅ **Stopping criteria** - `StopOnSubstrings` prevents over-generation
- ✅ **Answer extraction** - `extract_answer_from_generation()` cleans outputs

### Quick Import
```python
from Inference.inference_utils import (
    trim_answer,                      # Trim to N sentences
    format_rwku,                      # Format Q&A prompts
    StopOnSubstrings,                 # Stopping criteria class
    STOP_SEQUENCES,                   # Default stop strings
    extract_answer_from_generation,   # Extract clean answers
    get_blur_dataset,                 # Load BLUR/RWKU datasets
)
```

See **[UTILITIES_README.md](UTILITIES_README.md)** for detailed documentation.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers spacy
python -m spacy download en_core_web_sm
```

### 2. Run Inference

Generate responses from your unlearned model:

```bash
python inference_loop.py \
    --model_path path/to/unlearned_model \
    --data_path sample_questions.json \
    --output_path results/responses.json \
    --use_stopping_criteria \
    --trim_sentences 2 \
    --max_new_tokens 64 \
    --batch_size 4
```

**New flags:**
- `--use_stopping_criteria` - Stop generation at newlines/questions
- `--trim_sentences N` - Keep only N sentences in answers

### 3. Compute Metrics

Evaluate unlearning effectiveness:

```bash
python compute_metrics.py \
    --results_path results/responses.json \
    --output_path results/metrics.json
```

Metrics computed:
- **ROUGE-L** - Text similarity between original and unlearned answers
- **Perplexity Gap** - Model confidence on forget vs. retain sets
- **LLM-as-Judge** - Binary classification (forgotten vs. remembered)

---

## 📊 Evaluation Pipelines

### Option A: All-in-One Evaluation

Run inference + all metrics in one command:

```bash
python inference_loop.py \
    --model_path path/to/unlearned_model \
    --data_path sample_questions.json \
    --output_path results/responses.json \
    --compute_metrics \
    --judge_model_path path/to/judge_model \
    --perplexity_gap_model_path path/to/unlearned_model \
    --use_stopping_criteria \
    --trim_sentences 2
```

### Option B: Separate Evaluation Scripts

**Step 1:** Generate responses
```bash
python inference_loop.py \
    --model_path path/to/model \
    --data_path data.json \
    --output_path results/responses.json \
    --use_stopping_criteria
```

**Step 2:** Run Perplexity Gap evaluation
```bash
python run_perplexity_gap_evaluation.py \
    --model_path path/to/model \
    --results_path results/responses.json \
    --metrics_path results/perplexity_gap_metrics.json \
    --threshold 0.3
```

**Step 3:** Run LLM-as-Judge evaluation
```bash
python run_judge_evaluation.py \
    --results_path results/responses.json \
    --judge_model_path path/to/judge_model \
    --output_path results/responses_judged.json \
    --metrics_path results/judge_metrics.json
```

---

## 📋 Data Format

### Input Data (`sample_questions.json`)

```json
[
  {
    "prompt": "Question: What is the capital of France?\nAnswer:",
    "answer": "Paris",
    "split": "forget"
  },
  {
    "prompt": "Question: What is 2+2?\nAnswer:",
    "answer": "4",
    "split": "retain"
  }
]
```

**Fields:**
- `prompt` - Formatted question (use `format_rwku()` for consistency)
- `answer` - Ground truth answer
- `split` - Either `"forget"` or `"retain"`

### Output Format (`responses.json`)

```json
[
  {
    "question": "Question: What is the capital of France?\nAnswer:",
    "answer": "I don't know",
    "ground_truth": "Paris",
    "split": "forget",
    "judge_verdict": "YES",
    "perplexity_gap_score": 0.15
  }
]
```

---

## 📈 Metrics Explained

### 1. ROUGE-L F1
Measures text similarity between ground truth and generated answers.
- **Forget set:** Lower is better (≤ 0.3 indicates successful unlearning)
- **Retain set:** Higher is better (≥ 0.5 indicates knowledge retention)

### 2. Perplexity Gap
Measures model confidence based on conditional perplexity.
- **Forget set:** Lower is better (< 0.3 indicates model "forgot")
- **Retain set:** Higher is better (> 0.3 indicates model "remembers")

### 3. LLM-as-Judge
Binary classification using a fine-tuned judge model.
- **Forget set:** Higher "forgotten" rate is better (> 70%)
- **Retain set:** Higher "remembered" rate is better (> 70%)

---

## 🔧 Configuration

### Model Loading

**Standard HuggingFace model:**
```bash
python inference_loop.py --model_path meta-llama/Llama-2-7b-hf
```

**Local checkpoint:**
```bash
python inference_loop.py --model_path /path/to/local/checkpoint
```

**With wrapper (LlamaForBlur):**
```bash
python inference_loop.py --model_path path/to/model --use_wrapper
```

### Generation Parameters

```bash
--max_new_tokens 64         # Maximum tokens to generate
--temperature 1.0           # Sampling temperature (use 0.0 for greedy)
--batch_size 4              # Batch size for inference
--use_stopping_criteria     # Enable stopping criteria
--trim_sentences 2          # Trim answers to 2 sentences
```

---

## 🧪 Testing

Verify shared utilities work correctly:

```bash
cd Classification/Inference
python test_inference_utils.py
```

Expected output:
```
Testing trim_answer...
✓ trim_answer tests passed

Testing format_rwku...
✓ format_rwku test passed

Testing StopOnSubstrings...
✓ StopOnSubstrings tests passed

Testing extract_answer_from_generation...
✓ extract_answer_from_generation tests passed

ALL TESTS PASSED ✓
```

---

## 📖 Examples

### Example 1: Format a Question

```python
from Inference.inference_utils import format_rwku

example = {"text": "What is machine unlearning?"}
formatted = format_rwku(example)
print(formatted["text"])
# Output: "Question: What is machine unlearning?\nAnswer:"
```

### Example 2: Trim Long Answers

```python
from Inference.inference_utils import trim_answer

long_answer = "First. Second. Third. Fourth."
short = trim_answer(long_answer, max_sentences=2)
print(short)
# Output: "First. Second."
```

### Example 3: Use Stopping Criteria

```python
from Inference.inference_utils import StopOnSubstrings, STOP_SEQUENCES
from transformers import StoppingCriteriaList

stopping_criteria = StoppingCriteriaList([
    StopOnSubstrings(STOP_SEQUENCES, tokenizer=tokenizer)
])

output = model.generate(
    **inputs,
    max_new_tokens=50,
    stopping_criteria=stopping_criteria
)
```

---

## 🐛 Troubleshooting

### Issue: spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Issue: CUDA out of memory
- Reduce `--batch_size`
- Reduce `--max_new_tokens`
- Models are loaded with 4-bit quantization by default

### Issue: Import errors
Make sure you're running from the `Classification/` directory or that the parent directory is in `sys.path`.

---

## 📚 Additional Resources

- **[UTILITIES_README.md](UTILITIES_README.md)** - Detailed utilities documentation
- **[test_inference_utils.py](test_inference_utils.py)** - Test suite with examples
- **[sample_questions.json](sample_questions.json)** - Example input data
- **[sample_inference_results.json](sample_inference_results.json)** - Example output data

---

## 🎯 Best Practices

1. **Always use shared utilities** - Import from `inference_utils.py` for consistency
2. **Enable stopping criteria** - Use `--use_stopping_criteria` to prevent over-generation
3. **Trim answers** - Use `--trim_sentences 2` for concise outputs
4. **Format prompts consistently** - Use `format_rwku()` for all Q&A data
5. **Test before deployment** - Run `test_inference_utils.py` after changes

---

## 📝 Citation

If you use this code, please cite the SEMU paper:

```bibtex
@article{semu2024,
  title={SEMU: Selective Unlearning for Machine Learning},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 🤝 Contributing

When adding new inference or evaluation scripts:
1. Import and use utilities from `inference_utils.py`
2. Follow the established data format
3. Add tests to `test_inference_utils.py`
4. Update this README with new features

---

## 📞 Support

For issues or questions:
1. Check the [UTILITIES_README.md](UTILITIES_README.md) for detailed docs
2. Run tests: `python test_inference_utils.py`
3. Check example usage in existing scripts
