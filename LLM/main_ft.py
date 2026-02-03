import finetune
from models.model_llama import get_model
from mapped_datasets import *
from datasets import concatenate_datasets, DatasetDict


def main():
    model = get_model(load_in_4bit = True)
    dataset_trivia = get_trivia_qa_dataset()
    dataset_nq = get_nq_dataset()

    mixed_train = concatenate_datasets([
        dataset_trivia["train"],
        dataset_trivia["train"],  # TriviaQA x2
        dataset_nq["train"],      # NQ x1
    ]).shuffle(seed=42)
    mixed_dataset = DatasetDict({
        "train": mixed_train
    })  

    finetune.finetune_model(model, mixed_dataset, "./adapters/adapter-llama-2-7b-nq-trivia-qa")

if __name__ == "__main__":
    main()