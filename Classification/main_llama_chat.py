from models.model_llama import LlamaForBlur, get_model
import torch 

model = get_model()
model.eval()

prompt = "What causes Parkinson's disease?"
tokenizer = model.get_tokenizer()
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
