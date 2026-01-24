from transformers import Trainer, DataCollatorForLanguageModeling
from models.model_llama import LlamaForBlur
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from models.model_llama import LlamaForBlur 

def finetune_model(llama_for_blur: LlamaForBlur, dataset, output_dir = "./finetuned-qa"):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(llama_for_blur.model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,
        fp16=True,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit"
    )

    
    tokenizer = llama_for_blur.get_tokenizer()
    tokenized = get_tokenized_dataset(tokenizer, dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    trainer.train()

def get_tokenized_dataset(tokenizer, dataset):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )

    return dataset.map(tokenize, batched=True)