"""
Shared utilities for inference and evaluation.
Provides consistent dataset formatting, answer trimming, and stopping criteria.
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


def get_blur_dataset(dataset_name: str = "rwku", split: str = "retain"):
    """
    Load and format BLUR dataset with consistent prompting.
    
    Args:
        dataset_name: Name of the BLUR dataset (default: "rwku")
        split: Dataset split to load (default: "retain")
    
    Returns:
        Formatted dataset with prompts
    """
    from evaluation import BLUR
    
    blur = BLUR.get_BLUR_dataset(dataset_name, split)
    dataset = blur.dataset.map(format_rwku)
    return dataset


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
