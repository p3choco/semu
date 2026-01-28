import copy
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

    # Device setup
    # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    if args.seed:
        utils.setup_seed(args.seed)

    # === Load model & datasets ===
    model, train_dataset_full, retain_dataset, forget_dataset = utils.setup_model_dataset(args)

    # Move model to device
    model = get_model(finetune=args.finetune, train=args.train)
    print(model.model)
    # === Split retain / forget datasets ===


    forget_loader = DataLoader(forget_dataset["train"], batch_size=args.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset["train"], batch_size=args.batch_size, shuffle=True)
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

    criterion = nn.CrossEntropyLoss()  # can also use LabelSmoothing if needed
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
    unlearn_method = unlearn.get_unlearn_method(args.unlearn)
    unlearn_method(data_loaders, model, criterion, args)
    unlearn.save_unlearn_checkpoint(model, None, args)

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
