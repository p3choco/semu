# SEMU: Singular Value Decomposition for Efficient Machine Unlearning

**WARNING: This repository contains model outputs that may be offensive in nature.**

## Cite This Work
```
@article{sendera2025semu,
  title={SEMU: Singular Value Decomposition for Efficient Machine Unlearning},
  author={Sendera, Marcin and Struski, {\L}ukasz and Ksi{\k{a}}{\.z}ek, Kamil and Musiol, Kryspin and Tabor, Jacek and Rymarczyk, Dawid},
  journal={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## Run inference loop

### Generate responses and compute metrics

Basic run (response generation only):

```bash
python Classification/Inference/inference_loop.py \
  --model_path <path/to/model> \
  --data_path <path/to/data.json> \
  --output_path <path/to/output.json>
```

### Compute ROUGE, Perplexity gap, and LLM-as-Judge metrics

To compute all metrics after generation, use the following flags:

```bash
python Classification/Inference/inference_loop.py \
  --model_path <path/to/model> \
  --data_path <path/to/data.json> \
  --output_path <path/to/output.json> \
  --compute_metrics \
  --perplexity_gap_model_path <path/to/perplexity_gap_model> \
  --judge_model_path <path/to/judge_model>
```

- `--compute_metrics` – computes metrics on generated responses.
- `--perplexity_gap_model_path` – path to the model for Perplexity gap evaluation (can be the same as the generation model or different).
- `--judge_model_path` – path to the model for LLM-as-Judge evaluation.

#### Example

```bash
python Classification/Inference/inference_loop.py \
  --model_path models/llama2 \
  --data_path data/eval.json \
  --output_path results/llama2_responses.json \
  --compute_metrics \
  --perplexity_gap_model_path models/llama2 \
  --judge_model_path models/judge-llama2
```

#### Additional options

- `--perplexity_gap_threshold` – success threshold for Perplexity gap (default: 0.3)
- `--perplexity_gap_max_length` – max sequence length for Perplexity gap computation (default: 512)
- `--judge_batch_size` – batch size for LLM-as-Judge (default: 4)

---

### Compute metrics on an existing results file

If you already have a results file, you can compute metrics separately:

```bash
python Classification/Inference/compute_metrics.py \
  --results_path <path/to/output.json> \
  --perplexity_gap_model_path <path/to/perplexity_gap_model>
```

Add `--output_path <path/to/metrics.json>` to save metrics to a file.
