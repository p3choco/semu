"""
Run Perplexity gap (Membership Inference Attack) evaluation for unlearning.
Checks if model "remembers" forget set vs retain set.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Inference.compute_metrics import (
    load_results,
    run_perplexity_gap_evaluation,
    compute_metrics,
    print_results_table,
    save_metrics,
)


def load_model_for_perplexity_gap(model_path: str, device: str = "cuda"):
    """Load model for Perplexity gap evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model from: {model_path}")

    is_local = os.path.exists(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=is_local, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=is_local,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run Perplexity gap evaluation for unlearning")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to unlearned model")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to responses JSON file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save results with Perplexity gap scores")
    parser.add_argument("--metrics_path", type=str, default=None,
                        help="Path to save metrics")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Perplexity gap threshold for success")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length for Perplexity gap computation")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load results
    results = load_results(args.results_path)
    print(f"Loaded {len(results)} samples")

    # Separate forget and retain as lists of dicts
    forget_items = [item for item in results if item.get("split", "retain").lower() == "forget"]
    retain_items = [item for item in results if item.get("split", "retain").lower() == "retain"]

    print(f"Forget set: {len(forget_items)} | Retain set: {len(retain_items)}")

    # Load model
    model, tokenizer = load_model_for_perplexity_gap(args.model_path, args.device)
    device = next(model.parameters()).device

    # Run Perplexity gap
    print("\nRunning Perplexity gap evaluation...")
    perplexity_gap_results = run_perplexity_gap_evaluation(
        model, tokenizer,
        forget_items, retain_items,
        device=str(device),
        threshold=args.threshold,
        max_length=args.max_length,
    )

    # Compute full metrics
    metrics = compute_metrics(results, mia_results=perplexity_gap_results)
    print_results_table(metrics)

    if args.metrics_path:
        save_metrics(metrics, args.metrics_path)

    return metrics


if __name__ == "__main__":
    main()
