"""
Inference Pipeline for LLaMA models after unlearning.
Generates responses and saves them to a file for later evaluation.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import StoppingCriteriaList

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Inference.inference_utils import (
    StopOnSubstrings,
    STOP_SEQUENCES,
    extract_answer_from_generation,
    trim_answer,
)


def load_data(data_path: str) -> List[Dict]:
    """Load evaluation data from JSON or CSV file."""
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
    """Load model and tokenizer in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading model from: {model_path}")
    is_local = os.path.exists(model_path)
    
    if use_wrapper:
        from models.model_llama import LlamaForBlur
        
        class Args:
            arch = model_path
        
        model_wrapper = LlamaForBlur(Args())
        model = model_wrapper.model
        tokenizer = model_wrapper.get_tokenizer()
        
        if is_local:
            for ckpt_name in ["pytorch_model.bin", "model.safetensors"]:
                checkpoint_file = os.path.join(model_path, ckpt_name)
                if os.path.exists(checkpoint_file):
                    print(f"Loading checkpoint from {checkpoint_file}")
                    state_dict = torch.load(checkpoint_file, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                    break
    else:
        if not is_local and model_path.startswith('/'):
            raise ValueError(f"Local path '{model_path}' does not exist.")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=is_local, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # 4-bit quantization config
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
    dev = model.device if hasattr(model, 'device') else next(model.parameters()).device
    print(f"Model loaded. Device: {dev}")
    
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    batch_size: int = 4,
    use_stopping_criteria: bool = True,
    trim_sentences: int = 2,
) -> List[str]:
    """Run inference and return generated responses."""
    all_responses = []
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    # Setup stopping criteria
    stopping_criteria = None
    if use_stopping_criteria:
        stopping_criteria = StoppingCriteriaList([
            StopOnSubstrings(STOP_SEQUENCES, tokenizer=tokenizer)
        ])
    
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
                stopping_criteria=stopping_criteria,
            )
            
            for j, output in enumerate(outputs):
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                # Extract and clean answer using shared utility
                response = extract_answer_from_generation(decoded, trim_sentences=trim_sentences)
                all_responses.append(response)
    
    return all_responses


def save_results(results: List[Dict], output_path: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate responses with LLaMA model")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation data (JSON/CSV)")
    parser.add_argument("--output_path", type=str, default="results/responses.json",
                        help="Path to save generated responses")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_wrapper", action="store_true",
                        help="Use LlamaForBlur wrapper")
    parser.add_argument("--use_stopping_criteria", action="store_true",
                        help="Use stopping criteria to prevent over-generation")
    parser.add_argument("--trim_sentences", type=int, default=2,
                        help="Number of sentences to keep in answers")
    parser.add_argument("--compute_metrics", action="store_true",
                        help="Also compute metrics after generation")
    parser.add_argument("--judge_model_path", type=str, default=None,
                        help="Path to fine-tuned judge model (for LLM-as-Judge)")
    parser.add_argument("--judge_batch_size", type=int, default=4,
                        help="Batch size for judge model")
    parser.add_argument("--perplexity_gap_model_path", type=str, default=None,
                        help="Path to model for Perplexity gap evaluation")
    parser.add_argument("--perplexity_gap_threshold", type=float, default=0.3,
                        help="Threshold for Perplexity gap success")
    parser.add_argument("--perplexity_gap_max_length", type=int, default=512,
                        help="Max sequence length for Perplexity gap computation")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} samples")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.device, args.use_wrapper
    )
    
    # Run inference
    prompts = [item["prompt"] for item in data]
    print("\nGenerating responses...")
    responses = run_inference(
        model, tokenizer, prompts,
        args.max_new_tokens, args.temperature, args.batch_size,
        args.use_stopping_criteria, args.trim_sentences
    )
    
    # Build results
    results = []
    for item, response in zip(data, responses):
        results.append({
            "question": item["prompt"],
            "answer": response,
            "ground_truth": item["answer"],
            "split": item.get("split", "retain"),
        })
    
    # Save results
    save_results(results, args.output_path)

    # Optionally compute metrics
    if args.compute_metrics:
        from Inference.compute_metrics import compute_and_print_metrics, run_llm_judge, run_perplexity_gap_evaluation
        mia_results = None
        # Perplexity gap if model provided
        if args.perplexity_gap_model_path:
            from Inference.compute_metrics import run_perplexity_gap_evaluation
            model_pg, tokenizer_pg = load_model_and_tokenizer(args.perplexity_gap_model_path, args.device)
            forget_items = [item for item in results if item.get("split", "retain").lower() == "forget"]
            retain_items = [item for item in results if item.get("split", "retain").lower() == "retain"]
            mia_results = run_perplexity_gap_evaluation(
                model_pg, tokenizer_pg,
                forget_items, retain_items,
                device=args.device,
                threshold=args.perplexity_gap_threshold,
                max_length=args.perplexity_gap_max_length,
            )
        # Standard metrics (with Perplexity gap if calculated)
        compute_and_print_metrics(args.output_path, mia_results=mia_results)
        # LLM-as-Judge if judge_model_path provided
        if args.judge_model_path:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            is_local = os.path.exists(args.judge_model_path)
            judge_tokenizer = AutoTokenizer.from_pretrained(
                args.judge_model_path, local_files_only=is_local, trust_remote_code=True
            )
            if judge_tokenizer.pad_token is None:
                judge_tokenizer.pad_token = judge_tokenizer.eos_token
            judge_tokenizer.padding_side = "left"

            # 4-bit quantization config for judge model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            judge_model = AutoModelForCausalLM.from_pretrained(
                args.judge_model_path,
                quantization_config=bnb_config,
                device_map="auto" if args.device == "cuda" and torch.cuda.is_available() else None,
                local_files_only=is_local,
                trust_remote_code=True,
            )
            judge_model.eval()
            print("\nRunning LLM-as-Judge evaluation...")
            results_with_judge = run_llm_judge(
                results,
                judge_model,
                judge_tokenizer,
                batch_size=args.judge_batch_size
            )
            # Save results with judge verdicts
            judge_output_path = args.output_path.replace(".json", "_judged.json")
            save_results(results_with_judge, judge_output_path)
            # Print metrics for judged results (with Perplexity gap if calculated)
            compute_and_print_metrics(judge_output_path, mia_results=mia_results)

    return results

if __name__ == "__main__":
    main()
