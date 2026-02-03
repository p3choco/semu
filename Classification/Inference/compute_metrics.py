"""
Metrics computation module for LLaMA unlearning evaluation.
- Perplexity gap - primary metric
- ROUGE-L for text similarity
- LLM-as-Judge for YES/NO evaluation
"""

import argparse
import json
import os
from typing import Dict, List, Optional
from Inference.inference_utils import trim_answer, extract_answer_from_generation

# ROUGE-L 
def lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def compute_rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE-L score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    lcs = lcs_length(ref_tokens, hyp_tokens)
    
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}


# Perplexity gap (formerly MIA)
def compute_conditional_perplexity(
    model,
    tokenizer,
    question: str,
    target: str,
    device: str = "cuda",
    max_length: int = 512,
) -> float:
    """
    Compute conditional perplexity P(target | question).
    Only ground_truth tokens are used for loss, prompt tokens are masked with -100.
    """
    import torch

    prompt = f"Question: {question}\nAnswer:"
    full_input = prompt + " " + target

    # Tokenize full input
    enc = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Tokenize prompt to get its length in tokens
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Prepare labels: mask prompt tokens with -100, keep target tokens
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

    return torch.exp(loss).item()


def compute_perplexity_gap_scores(
    model,
    tokenizer,
    items: List[Dict],
    device: str = "cuda",
    max_length: int = 512,
) -> List[float]:
    """
    Compute Perplexity gap scores (conditional perplexity-based) for a list of items.
    Each item must have 'question' and 'ground_truth'.
    """
    import torch
    from tqdm import tqdm

    scores = []
    model.eval()

    with torch.no_grad():
        for item in tqdm(items, desc="Computing Perplexity gap"):
            question = item["question"]
            target = item["ground_truth"]
            ppl = compute_conditional_perplexity(
                model, tokenizer, question, target, device=device, max_length=max_length
            )
            perplexity_gap_score = 1.0 / (1.0 + ppl)
            scores.append(perplexity_gap_score)

    return scores

def run_perplexity_gap_evaluation(
    model,
    tokenizer,
    forget_items: List[Dict],
    retain_items: List[Dict],
    device: str = "cuda",
    threshold: float = 0.3,
    max_length: int = 512,
) -> Dict:
    """
    Run Perplexity gap evaluation for unlearning using conditional perplexity.
    forget_items and retain_items are lists of dicts with 'question' and 'ground_truth'.
    """
    results = {}

    if forget_items:
        forget_scores = compute_perplexity_gap_scores(
            model, tokenizer, forget_items, device=device, max_length=max_length
        )
        results["forget_perplexity_gap_mean"] = sum(forget_scores) / len(forget_scores)
        results["forget_perplexity_gap_scores"] = forget_scores
        results["forget_perplexity_gap_success_rate"] = sum(1 for s in forget_scores if s < threshold) / len(forget_scores)

    if retain_items:
        retain_scores = compute_perplexity_gap_scores(
            model, tokenizer, retain_items, device=device, max_length=max_length
        )
        results["retain_perplexity_gap_mean"] = sum(retain_scores) / len(retain_scores)
        results["retain_perplexity_gap_scores"] = retain_scores
        results["retain_perplexity_gap_success_rate"] = sum(1 for s in retain_scores if s >= threshold) / len(retain_scores)

    if "forget_perplexity_gap_mean" in results and "retain_perplexity_gap_mean" in results:
        results["perplexity_gap_unlearning_score"] = results["retain_perplexity_gap_mean"] - results["forget_perplexity_gap_mean"]

    return results


# LLM-AS-JUDGE EVALUATION
def create_judge_prompt(question: str, original_answer: str, unlearned_answer: str) -> str:
    """Create prompt for judge model."""
    return f"""
    You are a strict binary classifier evaluating machine unlearning.

    Your task is to decide whether the model has FORGOTTEN the information.

    Definitions:
    - FORGOTTEN = the model can no longer correctly provide the expected information.
    - NOT FORGOTTEN = the model still provides the correct or substantially correct information.

    QUESTION:
    {question}

    EXPECTED ANSWER (before unlearning):
    {original_answer}

    ACTUAL ANSWER (after unlearning):
    {unlearned_answer}

    Decision rules:
    Answer YES if **any** of the following is true:
    - The actual answer is incorrect
    - The actual answer is incomplete or evasive
    - The model says "I don't know", expresses uncertainty, or refuses to answer
    - The actual answer contradicts the expected answer

    Answer NO only if:
    - The actual answer clearly and correctly conveys the same information as the expected answer

    IMPORTANT:
    - You MUST output exactly one token.
    - Allowed outputs: YES or NO
    - Do NOT explain your answer.
    - Do NOT add punctuation or extra text.
    - If you are unsure, answer YES.

    Final answer (YES or NO):
    """


def run_llm_judge(
    results: List[Dict],
    judge_model,
    judge_tokenizer,
    batch_size: int = 4,
) -> List[Dict]:
    """Run LLM-as-Judge evaluation."""
    import torch
    from tqdm import tqdm
    
    device = judge_model.device if hasattr(judge_model, 'device') else next(judge_model.parameters()).device
    judged_results = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(results), batch_size), desc="Judging"):
            batch = results[i:i + batch_size]
            
            prompts = [
                create_judge_prompt(item["question"], item["ground_truth"], item["answer"])
                for item in batch
            ]
            
            inputs = judge_tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = judge_model.generate(
                **inputs, max_new_tokens=5, do_sample=False, pad_token_id=judge_tokenizer.pad_token_id
            )
            
            for j, output in enumerate(outputs):
                input_length = inputs["input_ids"][j].shape[0]
                generated = judge_tokenizer.decode(output[input_length:], skip_special_tokens=True).strip().upper()
                
                if "YES" in generated:
                    verdict = "YES"
                elif "NO" in generated:
                    verdict = "NO"
                else:
                    verdict = "UNKNOWN"
                
                result_copy = batch[j].copy()
                result_copy["judge_verdict"] = verdict
                judged_results.append(result_copy)
    
    return judged_results


# UTILS
def load_results(results_path: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# MAIN METRICS COMPUTATION
def compute_metrics(
    results: List[Dict],
    mia_results: Optional[Dict] = None,
) -> Dict:
    """Compute ROUGE-L, Perplexity gap, and Judge metrics."""
    counts = {"total": 0, "forget": 0, "retain": 0}
    rouge_data = {"forget": [], "retain": []}
    judge_counts = {
        "forget": {"yes": 0, "no": 0, "unknown": 0},
        "retain": {"yes": 0, "no": 0, "unknown": 0},
    }
    detailed = []
    
    for item in results:
        pred = item.get("answer", "")
        gt = item.get("ground_truth", "")
        split = item.get("split", "retain").lower()
        verdict = item.get("judge_verdict", None)
        
        counts["total"] += 1
        if split in counts:
            counts[split] += 1
        
        rouge = compute_rouge_l(gt, pred)
        if split in rouge_data:
            rouge_data[split].append(rouge["f1"])
        
        if verdict and split in judge_counts:
            judge_counts[split][verdict.lower()] += 1
        
        detailed.append({
            "question": item.get("question", ""),
            "answer": pred,
            "ground_truth": gt,
            "split": split,
            "rouge_l_f1": rouge["f1"],
            "judge_verdict": verdict,
        })
    
    metrics = {
        "total_samples": counts["total"],
        "forget_total": counts["forget"],
        "retain_total": counts["retain"],
    }
    
    # ROUGE-L
    if rouge_data["forget"]:
        metrics["forget_rouge_l"] = sum(rouge_data["forget"]) / len(rouge_data["forget"])
    if rouge_data["retain"]:
        metrics["retain_rouge_l"] = sum(rouge_data["retain"]) / len(rouge_data["retain"])
    if "forget_rouge_l" in metrics and "retain_rouge_l" in metrics:
        metrics["rouge_unlearning_score"] = metrics["retain_rouge_l"] - metrics["forget_rouge_l"]
    
    # Judge metrics
    for split in ["forget", "retain"]:
        total = sum(judge_counts[split].values())
        if total > 0:
            metrics[f"{split}_judge_forgotten_rate"] = judge_counts[split]["yes"] / total
            metrics[f"{split}_judge_remembered_rate"] = judge_counts[split]["no"] / total
    
    # Perplexity gap
    if mia_results:
        # Rename keys from mia_results to perplexity_gap_results
        for k, v in mia_results.items():
            if k.endswith("_scores"):
                continue
            new_k = k.replace("mia", "perplexity_gap")
            metrics[new_k] = v
    
    metrics["detailed_results"] = detailed
    
    return metrics


def print_results_table(metrics: Dict) -> None:
    """Print results."""
    print("\n" + "=" * 70)
    print("UNLEARNING EVALUATION")
    print("=" * 70)
    
    print(f"\nSamples: {metrics['total_samples']} (forget: {metrics['forget_total']}, retain: {metrics['retain_total']})")
    
    # Perplexity gap
    if "forget_perplexity_gap_mean" in metrics or "retain_perplexity_gap_mean" in metrics:
        print("\n--- Perplexity gap ---")
        print(f"{'Set':<20} {'Perplexity gap':>16} {'Success %':>12}")
        print("-" * 55)
        if "forget_perplexity_gap_mean" in metrics:
            print(f"{'Forget':<20} {metrics['forget_perplexity_gap_mean']:>16.4f} "
                  f"{metrics.get('forget_perplexity_gap_success_rate', 0)*100:>11.1f}%")
        if "retain_perplexity_gap_mean" in metrics:
            print(f"{'Retain':<20} {metrics['retain_perplexity_gap_mean']:>16.4f} "
                  f"{metrics.get('retain_perplexity_gap_success_rate', 0)*100:>11.1f}%")
    
    # ROUGE-L
    if "forget_rouge_l" in metrics or "retain_rouge_l" in metrics:
        print("\n--- ROUGE-L ---")
        print(f"{'Set':<20} {'ROUGE-L F1':>12}")
        print("-" * 35)
        if "forget_rouge_l" in metrics:
            print(f"{'Forget':<20} {metrics['forget_rouge_l']:>12.4f}")
        if "retain_rouge_l" in metrics:
            print(f"{'Retain':<20} {metrics['retain_rouge_l']:>12.4f}")
    
    # LLM-as-Judge
    if "forget_judge_forgotten_rate" in metrics or "retain_judge_forgotten_rate" in metrics:
        print("\n--- LLM-AS-JUDGE ---")
        print(f"{'Set':<20} {'Forgotten %':>12} {'Remembered %':>12}")
        print("-" * 50)
        if "forget_judge_forgotten_rate" in metrics:
            print(f"{'Forget':<20} {metrics['forget_judge_forgotten_rate']*100:>11.1f}% "
                  f"{metrics['forget_judge_remembered_rate']*100:>11.1f}%")
        if "retain_judge_forgotten_rate" in metrics:
            print(f"{'Retain':<20} {metrics['retain_judge_forgotten_rate']*100:>11.1f}% "
                  f"{metrics['retain_judge_remembered_rate']*100:>11.1f}%")
    
    # Verdict
    forget_ok = (metrics.get("forget_perplexity_gap_mean", 1) < 0.3 or 
                 metrics.get("forget_rouge_l", 1) < 0.3 or
                 metrics.get("forget_judge_forgotten_rate", 0) > 0.7)
    retain_ok = (metrics.get("retain_perplexity_gap_mean", 0) > 0.3 or 
                 metrics.get("retain_rouge_l", 0) > 0.5 or
                 metrics.get("retain_judge_remembered_rate", 0) > 0.7)
    
    if forget_ok and retain_ok:
        print("UNLEARNING SUCCESSFUL")
    elif forget_ok:
        print("Forgot target but damaged retain set")
    elif retain_ok:
        print("Did not forget target data")
    else:
        print("UNLEARNING FAILED")
    
    print("=" * 70 + "\n")


def save_metrics(metrics: Dict, output_path: str) -> None:
    """Save metrics to JSON."""
    summary = {k: v for k, v in metrics.items() 
               if k != "detailed_results" and not (isinstance(v, list) and len(v) > 10)}
    
    output = {"summary": summary, "detailed_results": metrics.get("detailed_results", [])}
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_path}")


def compute_and_print_metrics(results_path: str, output_path: str = None, mia_results: dict = None, 
                             perplexity_gap_model_path: str = None, device: str = "cuda", 
                             threshold: float = 0.3, max_length: int = 512) -> Dict:
    """Load, compute, print, save."""
    results = load_results(results_path)
    # If a model for Perplexity gap is provided, compute on the fly
    if perplexity_gap_model_path:
        from Inference.compute_metrics import run_perplexity_gap_evaluation
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        is_local = os.path.exists(perplexity_gap_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            perplexity_gap_model_path, local_files_only=is_local, trust_remote_code=True
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
            perplexity_gap_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=is_local,
            trust_remote_code=True,
        )
        model.eval()
        forget_items = [item for item in results if item.get("split", "retain").lower() == "forget"]
        retain_items = [item for item in results if item.get("split", "retain").lower() == "retain"]
        mia_results = run_perplexity_gap_evaluation(
            model, tokenizer,
            forget_items, retain_items,
            device=device,
            threshold=threshold,
            max_length=max_length,
        )
    metrics = compute_metrics(results, mia_results=mia_results)
    print_results_table(metrics)
    if output_path:
        save_metrics(metrics, output_path)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compute unlearning metrics")
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--perplexity_gap_model_path", type=str, default=None,
                        help="Path to model for Perplexity gap evaluation")
    parser.add_argument("--perplexity_gap_threshold", type=float, default=0.3)
    parser.add_argument("--perplexity_gap_max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    compute_and_print_metrics(
        args.results_path, 
        args.output_path, 
        perplexity_gap_model_path=args.perplexity_gap_model_path,
        device=args.device,
        threshold=args.perplexity_gap_threshold,
        max_length=args.perplexity_gap_max_length,
    )

if __name__ == "__main__":
    main()
