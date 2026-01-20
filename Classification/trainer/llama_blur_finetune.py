from evaluation import BLUR
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from models.model_llama import LlamaForBlur, get_model
from trl import SFTTrainer


def train_blur():
    blur = BLUR.get_BLUR_dataset("rwku", "retain")
    dataset = blur.dataset.map(format_rwku, remove_columns=blur.dataset["train"].column_names)
    
    model = get_model()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    trainer = SFTTrainer(
        model=model,
        tokenizer=model.get_tokenizer(),
        train_dataset=dataset["train"],
        max_seq_length=512,
        packing=True
    )

    trainer.train()



def format_rwku(example):
    instruction = (
        "Determine the relationship between the two sentences. "
        "Respond with entailment, neutral, or contradiction."
    )
    print(example)

    prompt = f"""Sentence A: {example['sentence1']}
Sentence B: {example['sentence2']}

Question: {instruction}
Answer:"""

    return {
        "messages": [
            {"role": "system", "content": "You are a careful language understanding expert."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": LABEL_MAP[example["label"]]}
        ]
    }
