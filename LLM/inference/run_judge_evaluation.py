"""
Run LLM-as-Judge evaluation on generated responses.
Uses a fine-tuned model to judge whether unlearning was successful.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.compute_metrics import (
    load_results,
    run_llm_judge,
    compute_metrics,
    print_results_table,
    save_metrics,
)


def load_judge_model(model_path: str, device: str = "cuda"):
    """Load the judge model (fine-tuned model)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading judge model from: {model_path}")

    is_local = os.path.exists(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        local_files_only=is_local,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    print(f"Judge model loaded. Device: {next(model.parameters()).device}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge on unlearning results")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to responses JSON file")
    parser.add_argument("--judge_model_path", type=str, required=True,
                        help="Path to fine-tuned judge model")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save judged results")
    parser.add_argument("--metrics_path", type=str, default=None,
                        help="Path to save computed metrics")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load results
    print(f"Loading results from: {args.results_path}")
    results = load_results(args.results_path)
    print(f"Loaded {len(results)} samples")
    
    # Load judge model
    judge_model, judge_tokenizer = load_judge_model(args.judge_model_path, args.device)
    
    # Run judge evaluation
    print("\nRunning LLM-as-Judge evaluation...")
    judged_results = run_llm_judge(
        results, 
        judge_model, 
        judge_tokenizer, 
        args.batch_size
    )
    
    # Save judged results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(judged_results, f, indent=2, ensure_ascii=False)
        print(f"Judged results saved to: {args.output_path}")
    
    # Compute and print metrics
    metrics = compute_metrics(judged_results)
    print_results_table(metrics)
    
    if args.metrics_path:
        save_metrics(metrics, args.metrics_path)
    
    return judged_results, metrics


if __name__ == "__main__":
    main()
