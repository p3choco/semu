from models.model_llama import LlamaForBlur, get_model
from trainer.llama_blur_finetune import get_blur_mapped
import torch 
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnQuestion(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = tokenizer.encode("Question:")

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-len(self.stop_ids):].tolist() == self.stop_ids


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
model = get_model()
model.eval()
tokenizer = model.get_tokenizer()

stopping_criteria = StoppingCriteriaList([
    StopOnQuestion(tokenizer)
])

with torch.no_grad():
    for question in blur["train"]:
        prompt = question["text"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            stopping_criteria=stopping_criteria
        )

        print(tokenizer.decode(output[0], skip_special_tokens=True))
