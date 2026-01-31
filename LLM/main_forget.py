import copy
import torch
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# import arg_parser
import utils
import unlearn
# from finetune import validate
from models.model_llama import get_model 
from loss import LLMQuestionOnlyUnlearningLoss
from prompt_dataset import BlurPromptDataset
from peft import LoraConfig, get_peft_model

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--unlearn", type=str, default="own_SVD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="rwku")
    parser.add_argument("--decreasing_lr", type=str, default="30")
    return parser.parse_args()

def main():
    args = parse_args()
    # torch.cuda.empty_cache()
    print(torch.cuda.device_count())
    
    # device = torch.cuda(0)
    # print(torch.cuda.get_device_name(0))
    # Device setup
    # args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    if args.seed:
        utils.setup_seed(args.seed)

    # === Load model & datasets ===
    
    model, tokenizer, train_dataset_full, retain_dataset, forget_dataset = utils.setup_model_dataset(args)

    # Move model to device
    # model = get_model(finetune=args.finetune, train=args.train)
    # print(model.model)
    # === Split retain / forget datasets ===
    # forget_dataset = BlurPromptDataset(forget_dataset['text'], )
    for name, param in model.named_parameters():
        print(name)
    def get_tokenized_dataset(tokenizer, dataset):
        def tokenize(example):
            return tokenizer(
                example["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        tokenized.set_format(type="torch")  # converts input_ids & attention_mask to tensors
        return tokenized


    forget_tokenized = get_tokenized_dataset(tokenizer, forget_dataset)
    retain_tokenized = get_tokenized_dataset(tokenizer, retain_dataset)
    print(forget_tokenized)
    forget_loader = DataLoader(forget_tokenized, batch_size=args.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_tokenized, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    print(f"Number of retain examples: {len(retain_dataset)}")
    print(f"Number of forget examples: {len(forget_dataset)}")

    data_loaders = OrderedDict(
        retain=retain_loader,
        forget=forget_loader,
        # val=val_loader,
        # test=test_loader
    )

    # criterion = nn.CrossEntropyLoss()  # can also use LabelSmoothing if needed
    criterion = LLMQuestionOnlyUnlearningLoss(model)


    config = LoraConfig(
        r=8,                       # rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, config)
    print(peft_model)
    print("peft")
    for name, module in peft_model.named_modules():
        print(name, type(module))
    evaluation_result = {}

    # === Load or resume checkpoint ===
    # if args.resume:
    #     checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)
    #     if checkpoint is not None:
    #         model, evaluation_result = checkpoint
    # else:
    #     checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    #     if "state_dict" in checkpoint:
    #         checkpoint = checkpoint["state_dict"]

    #     if args.unlearn != "retrain":
    #         model.load_state_dict(checkpoint, strict=False)
    #         print("Loaded model checkpoint for LLM unlearning")

    # === Perform unlearning ===
    # unlearn_method = unlearn.get_unlearn_method(args.unlearn)
    unlearn_method = unlearn.get_unlearn_method("own_SVD")
    unlearn_method(data_loaders, peft_model, criterion, args)
    # unlearn.save_unlearn_checkpoint(model, None, args)

    # === Evaluate model on all splits ===
    # for name, loader in data_loaders.items():
        # convert dataset if needed (tokenization)
        # utils.dataset_convert_to_test(loader.dataset, args)
        # acc_or_score = validate(loader, model, criterion, args)
        # evaluation_result[name] = acc_or_score
        # print(f"{name} evaluation: {acc_or_score}")

    # === Save final checkpoint ===
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
