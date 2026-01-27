from datasets import load_dataset

def get_trivia_qa_dataset():
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext")

    def preprocess(ex): 
        return {
            "text": f"Question: {ex['question']}\nAnswer: {ex['answer']['value']}"
        }
    
    mapped = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
    return mapped.filter(lambda x: x["text"] is not None)
