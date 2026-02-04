from models.model_llama import get_model
import torch 
from transformers import StoppingCriteria, StoppingCriteriaList
from evaluation import BLUR
import spacy

nlp = spacy.load("en_core_web_sm")

def trim_answer(answer, max_sentences=2):
    doc = nlp(answer)
    return " ".join(sent.text for sent in list(doc.sents)[:max_sentences])

def get_blur_mapped():
    blur = BLUR.get_BLUR_dataset("rwku", "retain")
    dataset = blur.dataset.map(format_rwku)
    return dataset

def format_rwku(example):
    question = example["text"].strip()
    prompt = f"Question: {question}\nAnswer:"
    return {"text": prompt}

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # dekodujemy tylko nowo wygenerowany fragment
        decoded_text = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True
        )

        for s in self.stop_strings:
            if s in decoded_text:
                return True
        return False

STOP_SEQUENCES = [
    "\nQuestion:",
    "\nInstructions:",
    "\n\n"
]

if __name__ == "__main__":

    if torch.cuda.is_available():
    # check GPU compute capability
        gpu = torch.cuda.get_device_properties(0)
        if gpu.major >= 7:
            device = torch.device("cuda")
            print(f"Using GPU: {gpu.name}")
        else:
            device = torch.device("cpu")
            print(f"GPU {gpu.name} unsupported, falling back to CPU")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU")

    blur = get_blur_mapped()
    model = get_model(finetune="trivia_qa")
    model.eval()
    tokenizer = model.get_tokenizer()

    stopping_criteria = StoppingCriteriaList([
        StopOnSubstrings(
            STOP_SEQUENCES,
            tokenizer=tokenizer
        )
    ])

    with torch.no_grad():
        for prompt in blur["train"][:]["text"]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                stopping_criteria=stopping_criteria,
                eos_token_id=tokenizer.eos_token_id
            )

            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            answer = decoded.split("Answer:")[-1].strip()
            answer = answer.split("\n")[0].strip()
            answer = trim_answer(answer, max_sentences=2)

            print()
            print(prompt)
            print(answer)