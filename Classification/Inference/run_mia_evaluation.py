"""
Run MIA (Membership Inference Attack) evaluation for unlearning.
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
    run_mia_evaluation,
    compute_metrics,
    print_results_table,
    save_metrics,
)


def load_model_for_mia(model_path: str, device: str = "cuda"):
    """Load model for MIA evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model from: {model_path}")
    
    is_local = os.path.exists(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=is_local, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
        local_files_only=is_local,
        trust_remote_code=True,
    )
    model.eval()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run MIA evaluation for unlearning")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to unlearned model")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to responses JSON file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save results with MIA scores")
    parser.add_argument("--metrics_path", type=str, default=None,
                        help="Path to save metrics")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="MIA threshold for success")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load results
    results = load_results(args.results_path)
    print(f"Loaded {len(results)} samples")
    
    # Separate forget and retain
    forget_texts = []
    retain_texts = []
    
    for item in results:
        text = f"{item['question']} {item['ground_truth']}"
        split = item.get("split", "retain").lower()
        
        if split == "forget":
            forget_texts.append(text)
        else:
            retain_texts.append(text)
    
    print(f"Forget set: {len(forget_texts)} | Retain set: {len(retain_texts)}")
    
    # Load model
    model, tokenizer = load_model_for_mia(args.model_path, args.device)
    device = next(model.parameters()).device
    
    # Run MIA
    print("\nRunning MIA evaluation...")
    mia_results = run_mia_evaluation(
        model, tokenizer,
        forget_texts, retain_texts,
        device=str(device),
        threshold=args.threshold,
    )
    
    # Compute full metrics
    metrics = compute_metrics(results, mia_results=mia_results)
    print_results_table(metrics)
    
    if args.metrics_path:
        save_metrics(metrics, args.metrics_path)
    
    return metrics


if __name__ == "__main__":
    main()
