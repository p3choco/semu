from datasets import load_dataset
import random

def get_nq_dataset():
    dataset = load_dataset("google-research-datasets/nq_open")

    def map_nq(example):
        answers = example["answer"]

        # Skip examples without answers
        if answers is None or len(answers) == 0:
            return None

        answer = random.choice(answers)

        return {
            "text": f"Question: {example['question']}\nAnswer: {answer}"
        }

    mapped = dataset.map(map_nq, remove_columns=dataset["train"].column_names)
    return mapped.filter(lambda x: x["text"] is not None)



