import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
class LlamaForBlur(nn.Module):
    def __init__(self, /, *, finetune=None, load_in_4bit = False, train=False):
        """
        Wrapper for LLaMA 2 optimized for BLUR unlearning/evaluation.
        """
        super(LlamaForBlur, self).__init__()

        self.model_name = "meta-llama/Llama-2-7b-hf"

        print(f"--> Initializing LLaMA model: {self.model_name}")
        self.config = AutoConfig.from_pretrained(self.model_name)
        bnb_config = None
        if load_in_4bit:
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
                device_map="auto",
            )
            
            if finetune and 'trivia_qa' in str(finetune).lower():
                self.model = PeftModel.from_pretrained(
                    self.model, "Skryg/llama2-7b-trivia-qa")
                    
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        self.model.config.use_cache = False
        if train:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.model.eval()

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        """
        Forward pass that behaves like a standard HF model.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False
        )

        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits

    def generate(self, *args, **kwargs):
        """
        CRITICAL: Pass generation calls to the internal model.
        Used by BLUR benchmark for evaluation.
        """
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