import torch
import utils
import unlearn
import argparse
from collections import OrderedDict
from torch.utils.data import DataLoader
from loss import LLMQuestionOnlyUnlearningLoss
from mapped_datasets import map_random, map_input_ids_to_labels, random_answers
from transformers import DataCollatorForLanguageModeling

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="UL")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--unlearn", type=str, default="own_SVD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="rwku")
    parser.add_argument("--decreasing_lr", type=str, default="30")
    parser.add_argument("--unlearn_lr", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--unlearn_epochs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=10)
    return parser.parse_args()

def get_tokenized_dataset(tokenizer, dataset):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")
    return tokenized

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.seed:
        utils.setup_seed(args.seed)

    model, tokenizer, _, retain_dataset, forget_dataset = utils.setup_model_dataset(args)
    model = model.to(device)


    forget_tokenized = map_input_ids_to_labels(get_tokenized_dataset(tokenizer, forget_dataset))
    retain_tokenized = get_tokenized_dataset(tokenizer, retain_dataset)

    forget_train = map_random(forget_dataset, random_answers())
    forget_train_tokenized = map_input_ids_to_labels(get_tokenized_dataset(tokenizer, forget_train))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  
    )


    forget_train_loader = DataLoader(forget_train_tokenized, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    forget_loader = DataLoader(forget_tokenized, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    retain_loader = DataLoader(retain_tokenized, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    print(f"Number of retain examples: {len(retain_dataset)}")
    print(f"Number of forget examples: {len(forget_dataset)}")

    data_loaders = OrderedDict(
        retain=retain_loader,
        forget=forget_loader,
        forget_train=forget_train_loader
    )

    criterion = LLMQuestionOnlyUnlearningLoss(model)

    evaluation_result = {}
    print("Pobieram unlearn method...")
    unlearn_method = unlearn.get_unlearn_method("own_SVD")
    print("... wchodzę do niej ...")
    unlearn_method(data_loaders, model, criterion, args)
    print(" ... i wychodzę.")
    # unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    torch.save(model, "UL/unlearned_model.llama")

if __name__ == "__main__":
    main()
