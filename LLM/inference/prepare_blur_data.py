"""
Prepare BLUR dataset for inference pipeline.
Converts BLUR dataset format to inference-compatible JSON format.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference_utils import get_blur_dataset, convert_blur_to_inference_format
from evaluation.BLUR import get_BLUR_dataset


def prepare_blur_for_inference(
    task: str = "rwku",
    variant: str = "retain",
    output_path: str = None,
):
    """
    Load BLUR dataset and save it in inference-compatible format.
    
    Args:
        task: BLUR task ('rwku', 'whp', 'tofu', 'wmdp')
        variant: Dataset variant ('forget', 'retain', etc.)
        output_path: Where to save the JSON file
    """
    print(f"Loading BLUR dataset: task={task}, variant={variant}")
    
    # Use the new utility function
    data = get_blur_dataset(task=task, variant=variant)
    
    print(f"Loaded {len(data)} samples")
    print(f"Split: {data[0]['split']} (from variant: {variant})")
    
    # Show sample
    print("\nSample item:")
    print(f"  Prompt: {data[0]['prompt'][:100]}...")
    print(f"  Answer: {data[0]['answer']}")
    print(f"  Split: {data[0]['split']}")
    
    # Save to JSON
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved to: {output_path}")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BLUR dataset for inference pipeline"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="rwku",
        choices=["rwku", "whp", "tofu", "wmdp"],
        help="BLUR task name",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="retain",
        choices=["forget", "retain", "paired_forget_retain", "D_hi", "D_mid", "D_lo"],
        help="Dataset variant",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON file path (default: results/{task}_{variant}.json)",
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        args.output_path = f"results/{args.task}_{args.variant}_data.json"
    
    # Prepare dataset
    try:
        data = prepare_blur_for_inference(
            task=args.task,
            variant=args.variant,
            output_path=args.output_path,
        )
        
        print("\n" + "=" * 60)
        print("DATASET READY FOR INFERENCE")
        print("=" * 60)
        print(f"\nTotal samples: {len(data)}")
        print(f"Task: {args.task}")
        print(f"Variant: {args.variant}")
        print(f"Output: {args.output_path}")
        print("\nNext steps:")
        print(f"  python inference_loop.py \\")
        print(f"      --data_path {args.output_path} \\")
        print(f"      --model_path path/to/model \\")
        print(f"      --output_path results/responses.json \\")
        print(f"      --use_stopping_criteria \\")
        print(f"      --trim_sentences 2")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
