import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class LlamaForBlur(nn.Module):
    def __init__(self, /, *, train=False, num_train_layers=2):
        super(LlamaForBlur, self).__init__()

        self.model_name = "meta-llama/Llama-2-7b-hf"
        print(f"--> Initializing LLaMA model: {self.model_name}")
        self.config = AutoConfig.from_pretrained(self.model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        try:          
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=None,
            ).to("cuda:0")
            print(f"--> Model {self.model_name} loaded successfully.")       
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        base = self.model.model
        total_layers = len(base.layers)
        first_train_layer = total_layers-num_train_layers

        target_modules = []
        for i in range(first_train_layer, total_layers):
            target_modules.extend(
                [
                    f"layers.{i}.mlp.down_proj",
                    f"layers.{i}.mlp.up_proj",
                    f"layers.{i}.mlp.gate_proj",
                    f"layers.{i}.self_attn.q_proj",
                    f"layers.{i}.self_attn.k_proj",
                    f"layers.{i}.self_attn.v_proj",
                    f"layers.{i}.self_attn.o_proj"
                ]
            )

        LORA_R = 8
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        lora_config = LoraConfig(
            r = LORA_R,
            lora_alpha = LORA_ALPHA,
            lora_dropout = LORA_DROPOUT,
            target_modules = target_modules,
            bias = "none",
            task_type = "CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.num_train_layers = num_train_layers
        self.model.changed_layer_prefixes = [
            f"model.model.layers.{i}"
            for i in range(first_train_layer, total_layers)
        ]
        print(f"--> Applied LoRA with {len(target_modules)} target modules")
        
        self.model.config.use_cache = False
        if train:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            self.model.train()
        else:
            self.model.eval()

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.get_tokenizer().save_pretrained(path)

def get_model(**kwargs):
    return LlamaForBlur(**kwargs)