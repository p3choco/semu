"""
Shared utilities for inference and evaluation.
Provides consistent dataset formatting, answer trimming, and stopping criteria.

Key Functions:
- trim_answer: Trim answers to N sentences
- format_rwku: Format Q&A prompts consistently
- StopOnSubstrings: Stopping criteria for generation
- get_blur_dataset: Load BLUR dataset in inference-compatible format
- convert_blur_to_inference_format: Convert BLUR items to inference format
"""

import spacy
from transformers import StoppingCriteria

# Load spacy model for sentence tokenization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def trim_answer(answer: str, max_sentences: int = 2) -> str:
    """
    Trim answer to a maximum number of sentences using spacy.
    
    Args:
        answer: The generated answer text
        max_sentences: Maximum number of sentences to keep (default: 2)
    
    Returns:
        Trimmed answer containing at most max_sentences sentences
    """
    doc = nlp(answer)
    sentences = list(doc.sents)[:max_sentences]
    return " ".join(sent.text for sent in sentences)


def format_rwku(example: dict) -> dict:
    """
    Format RWKU dataset examples with consistent prompt structure.
    
    Args:
        example: Dictionary with 'text' field containing the question
    
    Returns:
        Dictionary with formatted 'text' field as "Question: {question}\nAnswer:"
    """
    question = example["text"].strip()
    prompt = f"Question: {question}\nAnswer:"
    return {"text": prompt}


class StopOnSubstrings(StoppingCriteria):
    """
    Stopping criteria for text generation that stops when specific substrings are encountered.
    Useful for preventing the model from generating beyond the expected answer format.
    """
    
    def __init__(self, stop_strings: list, tokenizer):
        """
        Initialize stopping criteria with list of stop strings.
        
        Args:
            stop_strings: List of strings that trigger stopping
            tokenizer: Tokenizer for decoding generated tokens
        """
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """
        Check if any stop string is present in the decoded text.
        
        Args:
            input_ids: Generated token IDs
            scores: Generation scores
        
        Returns:
            True if generation should stop, False otherwise
        """
        # Decode the generated text
        decoded_text = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True
        )
        
        # Check if any stop string is present
        for stop_string in self.stop_strings:
            if stop_string in decoded_text:
                return True
        return False


# Default stop sequences for Q&A format
STOP_SEQUENCES = [
    "\nQuestion:",
    "\nInstructions:",
    "\n\n"
]


def get_blur_dataset(task: str = "rwku", variant: str = "retain"):
    """
    Load and format BLUR dataset with consistent prompting.
    Compatible with inference pipeline format.
    
    Args:
        task: BLUR task name - 'rwku' | 'whp' | 'tofu' | 'wmdp' (default: "rwku")
        variant: Dataset variant - 'forget' | 'retain' | 'paired_forget_retain' | 'D_hi' | 'D_mid' | 'D_lo' (default: "retain")
    
    Returns:
        List of dictionaries with 'prompt', 'answer', and 'split' fields
        Ready for use with inference_loop.py
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from evaluation.BLUR import get_BLUR_dataset
    
    # Load BLUR dataset
    blur_dataset = get_BLUR_dataset(task, variant)
    
    # Convert to inference-compatible format
    formatted_data = []
    for i in range(len(blur_dataset)):
        item = blur_dataset[i]
        
        # Format the prompt with Q&A structure
        text = item["prompt"]
        if not text.startswith("Question:"):
            formatted_prompt = f"Question: {text}\nAnswer:"
        else:
            formatted_prompt = text
        
        # Map variant to split (forget/retain)
        split = "forget" if "forget" in variant.lower() else "retain"
        
        formatted_data.append({
            "prompt": formatted_prompt,
            "answer": "",  # Ground truth - will be filled during evaluation or can be extracted from dataset if available
            "split": split,
            "task": item.get("task", task),
            "variant": item.get("variant", variant),
        })
    
    return formatted_data


def convert_blur_to_inference_format(blur_dataset, ground_truths=None):
    """
    Convert BLUR dataset items to inference pipeline format.
    
    Args:
        blur_dataset: BLUR dataset instance or list of items from BLUR
        ground_truths: Optional list of ground truth answers (same length as dataset)
    
    Returns:
        List of dictionaries compatible with inference_loop.py format
    """
    formatted_data = []
    
    # Handle both Dataset object and list
    if hasattr(blur_dataset, '__len__') and hasattr(blur_dataset, '__getitem__'):
        items = [blur_dataset[i] for i in range(len(blur_dataset))]
    else:
        items = blur_dataset
    
    for idx, item in enumerate(items):
        # Extract prompt
        prompt = item.get("prompt", item.get("text", ""))
        
        # Format with Q&A structure if not already formatted
        if not prompt.startswith("Question:"):
            formatted_prompt = f"Question: {prompt}\nAnswer:"
        else:
            formatted_prompt = prompt
        
        # Determine split from variant
        variant = item.get("variant", "retain")
        split = "forget" if "forget" in str(variant).lower() else "retain"
        
        # Get ground truth if available
        answer = ""
        if ground_truths and idx < len(ground_truths):
            answer = ground_truths[idx]
        elif "answer" in item:
            answer = item["answer"]
        
        formatted_data.append({
            "prompt": formatted_prompt,
            "answer": answer,
            "split": split,
            "task": item.get("task", "unknown"),
        })
    
    return formatted_data


def extract_answer_from_generation(decoded_text: str, trim_sentences: int = 2) -> str:
    """
    Extract and clean the answer from generated text.
    
    Args:
        decoded_text: Full generated text including prompt
        trim_sentences: Number of sentences to keep in the answer
    
    Returns:
        Cleaned and trimmed answer
    """
    # Extract text after "Answer:"
    answer = decoded_text.split("Answer:")[-1].strip()
    
    # Take only the first line (stop at newline)
    answer = answer.split("\n")[0].strip()
    
    # Trim to specified number of sentences
    if trim_sentences > 0:
        answer = trim_answer(answer, max_sentences=trim_sentences)
    
    return answer
