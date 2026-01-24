"""
Metrics computation module for LLaMA unlearning evaluation.
- MIA (Membership Inference Attack) - primary metric
- ROUGE-L for text similarity
- LLM-as-Judge for YES/NO evaluation
"""

import argparse
import json
import os
from typing import Dict, List, Optional


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


# MIA 
def compute_perplexity(model, tokenizer, text: str, device: str = "cuda") -> float:
    """Compute perplexity of text under the model."""
    import torch
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    return torch.exp(loss).item()


def compute_mia_scores(model, tokenizer, texts: List[str], device: str = "cuda") -> List[float]:
    """Compute MIA scores (perplexity-based)."""
    import torch
    from tqdm import tqdm
    
    scores = []
    model.eval()
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing MIA"):
            ppl = compute_perplexity(model, tokenizer, text, device)
            mia_score = 1.0 / (1.0 + ppl)
            scores.append(mia_score)
    
    return scores


def run_mia_evaluation(
    model,
    tokenizer,
    forget_texts: List[str],
    retain_texts: List[str],
    device: str = "cuda",
    threshold: float = 0.3,
) -> Dict:
    """Run MIA evaluation."""
    results = {}
    
    if forget_texts:
        forget_scores = compute_mia_scores(model, tokenizer, forget_texts, device)
        results["forget_mia_mean"] = sum(forget_scores) / len(forget_scores)
        results["forget_mia_scores"] = forget_scores
        results["forget_mia_success_rate"] = sum(1 for s in forget_scores if s < threshold) / len(forget_scores)
    
    if retain_texts:
        retain_scores = compute_mia_scores(model, tokenizer, retain_texts, device)
        results["retain_mia_mean"] = sum(retain_scores) / len(retain_scores)
        results["retain_mia_scores"] = retain_scores
        results["retain_mia_success_rate"] = sum(1 for s in retain_scores if s >= threshold) / len(retain_scores)
    
    if "forget_mia_mean" in results and "retain_mia_mean" in results:
        results["mia_unlearning_score"] = results["retain_mia_mean"] - results["forget_mia_mean"]
    
    return results


# LLM-AS-JUDGE EVALUATION
def create_judge_prompt(question: str, original_answer: str, unlearned_answer: str) -> str:
    """Create prompt for judge model."""
    return f"""You are evaluating whether a model has successfully forgotten specific information.

QUESTION: {question}

EXPECTED ANSWER (before unlearning): {original_answer}

ACTUAL ANSWER (after unlearning): {unlearned_answer}

Has the model forgotten the information? Answer YES if:
- The answer is wrong or different
- The model says "I don't know"
- The model refuses to answer

Answer with only: YES or NO

Your answer:"""


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
    """Compute ROUGE-L, MIA, and Judge metrics."""
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
    
    # MIA
    if mia_results:
        metrics.update({k: v for k, v in mia_results.items() if not k.endswith("_scores")})
    
    metrics["detailed_results"] = detailed
    
    return metrics


def print_results_table(metrics: Dict) -> None:
    """Print results."""
    print("\n" + "=" * 70)
    print("UNLEARNING EVALUATION")
    print("=" * 70)
    
    print(f"\nSamples: {metrics['total_samples']} (forget: {metrics['forget_total']}, retain: {metrics['retain_total']})")
    
    # MIA
    if "forget_mia_mean" in metrics or "retain_mia_mean" in metrics:
        print("\n--- MIA ---")
        print(f"{'Set':<20} {'MIA Score':>12} {'Success %':>12}")
        print("-" * 50)
        if "forget_mia_mean" in metrics:
            print(f"{'Forget':<20} {metrics['forget_mia_mean']:>12.4f} "
                  f"{metrics.get('forget_mia_success_rate', 0)*100:>11.1f}%")
        if "retain_mia_mean" in metrics:
            print(f"{'Retain':<20} {metrics['retain_mia_mean']:>12.4f} "
                  f"{metrics.get('retain_mia_success_rate', 0)*100:>11.1f}%")
    
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
    print("\n--- VERDICT ---")
    forget_ok = (metrics.get("forget_mia_mean", 1) < 0.3 or 
                 metrics.get("forget_rouge_l", 1) < 0.3 or
                 metrics.get("forget_judge_forgotten_rate", 0) > 0.7)
    retain_ok = (metrics.get("retain_mia_mean", 0) > 0.3 or 
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


def compute_and_print_metrics(results_path: str, output_path: str = None) -> Dict:
    """Load, compute, print, save."""
    results = load_results(results_path)
    metrics = compute_metrics(results)
    print_results_table(metrics)
    if output_path:
        save_metrics(metrics, output_path)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute unlearning metrics")
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    
    compute_and_print_metrics(args.results_path, args.output_path)


if __name__ == "__main__":
    main()
