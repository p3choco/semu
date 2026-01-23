"""
Inference and Evaluation Pipeline for LLaMA models after unlearning.
Computes exact match, forget accuracy, and retain accuracy.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_data(data_path: str) -> List[Dict]:
    """
    Load evaluation data from JSON or CSV file.
    Expected fields: prompt, answer, split (forget/retain)
    """
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_path.endswith('.csv'):
        import csv
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return data


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    use_wrapper: bool = False,
):
    """
    Load model and tokenizer in eval mode.
    
    Args:
        model_path: Path to local model or HuggingFace model ID
        device: Device to use
        use_wrapper: If True, use LlamaForBlur wrapper from models/
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model from: {model_path}")
    
    is_local = os.path.exists(model_path)
    
    if use_wrapper:
        # Use the project's LlamaForBlur wrapper
        from models.model_llama import LlamaForBlur
        
        class Args:
            arch = model_path  # e.g., "7b" or "13b"
        
        model_wrapper = LlamaForBlur(Args())
        model = model_wrapper.model
        tokenizer = model_wrapper.get_tokenizer()
        
        # Load checkpoint if provided and exists
        if is_local:
            checkpoint_file = os.path.join(model_path, "pytorch_model.bin")
            if not os.path.exists(checkpoint_file):
                checkpoint_file = os.path.join(model_path, "model.safetensors")
            
            if os.path.exists(checkpoint_file):
                print(f"Loading checkpoint from {checkpoint_file}")
                state_dict = torch.load(checkpoint_file, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
    else:
        # Direct loading from HuggingFace or local path
        if not is_local and model_path.startswith('/'):
            raise ValueError(
                f"Local path '{model_path}' does not exist.\n"
                f"Please provide either:\n"
                f"  1. A valid local path to your model\n"
                f"  2. A HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b-hf')"
            )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=is_local,
            trust_remote_code=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Determine dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
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
    
    if hasattr(model, 'device'):
        dev = model.device
    else:
        dev = next(model.parameters()).device
    print(f"Model loaded successfully. Device: {dev}")
    
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    batch_size: int = 4,
) -> List[str]:
    """
    Run inference on a list of prompts and return generated responses.
    """
    all_responses = []
    
    # Get model device
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode only the newly generated tokens
            for j, output in enumerate(outputs):
                input_length = inputs["input_ids"][j].shape[0]
                generated_tokens = output[input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_responses.append(response.strip())
    
    return all_responses


def parse_answer(response: str, answer_type: str = "auto") -> str:
    """
    Parse the model's response to extract the actual answer.
    """
    response = response.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        r"^(The answer is|Answer:|A:|Response:)\s*",
        r"^(I think|I believe|It is|It's)\s+",
    ]
    for pattern in prefixes_to_remove:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE)
    
    if answer_type == "auto":
        answer_type = _detect_answer_type(response)
    
    if answer_type == "yesno":
        match = re.search(r'\b(yes|no|true|false)\b', response, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
            return "yes" if answer in ["yes", "true"] else "no"
        return response.split()[0].lower() if response else ""
    
    elif answer_type == "date":
        date_patterns = [
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
            r'\b(\d{4})\b',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        return response.strip()
    
    elif answer_type == "number":
        match = re.search(r'[-+]?\d*\.?\d+', response)
        if match:
            return match.group(0)
        return response.strip()
    
    elif answer_type == "entity":
        response = re.split(r'[.!?\n]', response)[0]
        response = response.strip().strip('.,!?:;')
        return response
    
    first_line = response.split('\n')[0].strip()
    return first_line


def _detect_answer_type(response: str) -> str:
    """Auto-detect the type of answer based on content."""
    response_lower = response.lower()
    
    if re.match(r'^(yes|no|true|false)\b', response_lower):
        return "yesno"
    if re.search(r'\b\d{4}\b', response) and re.search(
        r'(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}[-/])', 
        response_lower
    ):
        return "date"
    if re.match(r'^[-+]?\d+\.?\d*$', response.strip()):
        return "number"
    
    return "entity"


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = ' '.join(answer.split())
    return answer


def compute_metrics(
    predictions: List[str],
    ground_truths: List[str],
    splits: List[str],
) -> Dict:
    """
    Compute evaluation metrics: exact match, forget accuracy, retain accuracy.
    """
    assert len(predictions) == len(ground_truths) == len(splits)
    
    results = {
        "total": {"correct": 0, "total": 0},
        "forget": {"correct": 0, "total": 0},
        "retain": {"correct": 0, "total": 0},
    }
    
    detailed_results = []
    
    for pred, gt, split in zip(predictions, ground_truths, splits):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)
        
        is_correct = pred_norm == gt_norm
        
        results["total"]["total"] += 1
        results["total"]["correct"] += int(is_correct)
        
        split_key = split.lower()
        if split_key in results:
            results[split_key]["total"] += 1
            results[split_key]["correct"] += int(is_correct)
        
        detailed_results.append({
            "prediction": pred,
            "ground_truth": gt,
            "split": split,
            "correct": is_correct,
        })
    
    metrics = {}
    
    if results["total"]["total"] > 0:
        metrics["exact_match_accuracy"] = results["total"]["correct"] / results["total"]["total"]
    else:
        metrics["exact_match_accuracy"] = 0.0
    
    if results["forget"]["total"] > 0:
        metrics["forget_accuracy"] = results["forget"]["correct"] / results["forget"]["total"]
        metrics["forget_total"] = results["forget"]["total"]
    else:
        metrics["forget_accuracy"] = None
        metrics["forget_total"] = 0
    
    if results["retain"]["total"] > 0:
        metrics["retain_accuracy"] = results["retain"]["correct"] / results["retain"]["total"]
        metrics["retain_total"] = results["retain"]["total"]
    else:
        metrics["retain_accuracy"] = None
        metrics["retain_total"] = 0
    
    metrics["total_samples"] = results["total"]["total"]
    metrics["detailed_results"] = detailed_results
    
    return metrics


def print_results_table(metrics: Dict) -> None:
    """Print results in a readable table format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    headers = ["Metric", "Value", "Samples"]
    rows = [
        ["Exact Match Accuracy", f"{metrics['exact_match_accuracy']:.4f}", str(metrics['total_samples'])],
    ]
    
    if metrics["forget_accuracy"] is not None:
        rows.append(["Forget Accuracy", f"{metrics['forget_accuracy']:.4f}", str(metrics['forget_total'])])
    
    if metrics["retain_accuracy"] is not None:
        rows.append(["Retain Accuracy", f"{metrics['retain_accuracy']:.4f}", str(metrics['retain_total'])])
    
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(3)]
    
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)))
    
    print("=" * 60 + "\n")


def save_results(metrics: Dict, output_path: str) -> None:
    """Save results to JSON file."""
    summary = {k: v for k, v in metrics.items() if k != "detailed_results"}
    
    output = {
        "summary": summary,
        "detailed_results": metrics.get("detailed_results", []),
    }
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA model after unlearning")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Path to model checkpoint or HuggingFace model ID")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation data (JSON/CSV)")
    parser.add_argument("--output_path", type=str, default="results/eval_results.json",
                        help="Path to save results")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--answer_type", type=str, default="auto",
                        choices=["auto", "entity", "yesno", "date", "number"],
                        help="Type of expected answers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--use_wrapper", action="store_true",
                        help="Use LlamaForBlur wrapper from models/")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} samples")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, 
        args.device,
        use_wrapper=args.use_wrapper,
    )
    
    # Extract prompts, answers, and splits
    prompts = [item["prompt"] for item in data]
    ground_truths = [item["answer"] for item in data]
    splits = [item.get("split", "retain") for item in data]
    
    # Run inference
    print("\nRunning inference...")
    responses = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )
    
    # Parse answers
    print("Parsing answers...")
    predictions = [parse_answer(resp, args.answer_type) for resp in responses]
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(predictions, ground_truths, splits)
    
    # Print and save results
    print_results_table(metrics)
    save_results(metrics, args.output_path)
    
    return metrics


if __name__ == "__main__":
    main()
