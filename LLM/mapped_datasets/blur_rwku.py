from evaluation import BLUR

def get_blur_mapped():
    blur = BLUR.get_BLUR_dataset("rwku", "retain")
    dataset = blur.dataset.map(format_rwku, remove_columns=blur.dataset["train"].column_names)
    return dataset



def format_rwku(example):
    print(example)
    instruction = (
        """Answer ONLY the question below. Answer the question correctly. Finish after answering the question."""
    )

    prompt = f"""Instructions: {instruction}
Question: {example['text']}
Answer:"""

    return {"text": prompt}
