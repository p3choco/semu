import torch
import utils
import unlearn
from collections import OrderedDict
from torch.utils.data import DataLoader
from loss import LLMCrossEntropyLoss
from mapped_datasets import map_random, map_input_ids_to_labels, random_answers
from transformers import DataCollatorForLanguageModeling
from arg_parser import parse_args

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

    criterion = LLMCrossEntropyLoss(model)

    unlearn_method = unlearn.get_unlearn_method("own_SVD")
    unlearn_method(data_loaders, model, criterion, args)

    torch.save(model, args.save_dir + "/unlearned_model.llama")

if __name__ == "__main__":
    main()
