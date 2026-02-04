import random

def map_random(dataset, answer_pool):
    
    SEED = 1234
    POOL_SIZE = len(answer_pool)
        
    random.seed(SEED)

    def add_random_answer(example):
        idx = random.randint(0, POOL_SIZE -1)
        return { "text": example["text"] + " " + answer_pool[idx] }


    dataset = dataset.map(
        add_random_answer,
        desc="Adding random answers",
    )
    return dataset

def map_input_ids_to_labels(dataset):
    def copy_input_ids_to_labels(example):
        example["labels"] = example["input_ids"]
        return example

    dataset = dataset.map(copy_input_ids_to_labels)
    return dataset