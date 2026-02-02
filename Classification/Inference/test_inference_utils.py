"""
Test script for inference utilities.
Verifies that trim_answer, format_rwku, and StopOnSubstrings work correctly.
"""

import torch
from transformers import AutoTokenizer, StoppingCriteriaList

from inference_utils import (
    trim_answer,
    format_rwku,
    StopOnSubstrings,
    STOP_SEQUENCES,
    extract_answer_from_generation,
)


def test_trim_answer():
    """Test trim_answer function."""
    print("Testing trim_answer...")
    
    # Test single sentence
    answer1 = "This is the first sentence."
    result1 = trim_answer(answer1, max_sentences=2)
    print(f"Input: {answer1}")
    print(f"Output: {result1}")
    assert result1 == "This is the first sentence."
    
    # Test multiple sentences
    answer2 = "First sentence here. Second sentence here. Third sentence here."
    result2 = trim_answer(answer2, max_sentences=2)
    print(f"\nInput: {answer2}")
    print(f"Output: {result2}")
    assert "First sentence" in result2 and "Second sentence" in result2
    assert "Third sentence" not in result2
    
    print("✓ trim_answer tests passed\n")


def test_format_rwku():
    """Test format_rwku function."""
    print("Testing format_rwku...")
    
    example = {"text": "What is the capital of France?"}
    result = format_rwku(example)
    
    expected = "Question: What is the capital of France?\nAnswer:"
    print(f"Input: {example}")
    print(f"Output: {result}")
    assert result["text"] == expected
    
    print("✓ format_rwku test passed\n")


def test_stop_on_substrings():
    """Test StopOnSubstrings stopping criteria."""
    print("Testing StopOnSubstrings...")
    
    # Use a simple tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create stopping criteria
    stopping_criteria = StopOnSubstrings(STOP_SEQUENCES, tokenizer)
    
    # Test case 1: Should NOT stop
    text1 = "This is a normal answer"
    tokens1 = tokenizer.encode(text1, return_tensors="pt")
    result1 = stopping_criteria(tokens1, None)
    print(f"Text: '{text1}' -> Stop: {result1}")
    assert result1 == False
    
    # Test case 2: Should stop (contains \nQuestion:)
    text2 = "This is an answer\nQuestion: What is next?"
    tokens2 = tokenizer.encode(text2, return_tensors="pt")
    result2 = stopping_criteria(tokens2, None)
    print(f"Text: '{text2}' -> Stop: {result2}")
    assert result2 == True
    
    print("✓ StopOnSubstrings tests passed\n")


def test_extract_answer():
    """Test extract_answer_from_generation function."""
    print("Testing extract_answer_from_generation...")
    
    # Test case 1: Normal generation
    gen1 = "Question: What is 2+2?\nAnswer: The answer is 4. This is correct."
    result1 = extract_answer_from_generation(gen1, trim_sentences=2)
    print(f"Input: {gen1}")
    print(f"Output: {result1}")
    assert "4" in result1
    
    # Test case 2: Generation with newline
    gen2 = "Question: Who?\nAnswer: John Smith\nQuestion: Next?"
    result2 = extract_answer_from_generation(gen2, trim_sentences=1)
    print(f"\nInput: {gen2}")
    print(f"Output: {result2}")
    assert "John Smith" in result2
    assert "Next" not in result2
    
    print("✓ extract_answer_from_generation tests passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("INFERENCE UTILITIES TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_trim_answer()
        test_format_rwku()
        test_stop_on_substrings()
        test_extract_answer()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
