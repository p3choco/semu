import torch
import arg_parser
import trainer.llama_blur_finetune as finetune
from models.model_llama import get_model
from mapped_datasets import *

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    model = get_model()
    dataset = get_trivia_qa_dataset()
    finetune.finetune_model(model, dataset, "./llama-2-7b-trivia-qa")

if __name__ == "__main__":
    main()