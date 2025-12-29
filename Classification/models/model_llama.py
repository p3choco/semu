import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


class LlamaForBlur(nn.Module):
    def __init__(self, args):
        """
        Wrapper for LLaMA 2 optimized for BLUR unlearning/evaluation.
        """
        super(LlamaForBlur, self).__init__()

        if hasattr(args, 'arch') and '13b' in args.arch.lower():
            self.model_name = "meta-llama/Llama-2-13b-hf"
        else:
            self.model_name = "meta-llama/Llama-2-7b-hf"

        print(f"--> Initializing LLaMA model: {self.model_name}")

        self.config = AutoConfig.from_pretrained(self.model_name)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

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


def get_model(args):
    return LlamaForBlur(args)