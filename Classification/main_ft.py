import torch
import arg_parser

import trainer.llama_blur_finetune as finetune

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    arr = finetune.train_blur()
    # args.save_dir = Path(args.save_dir) / datetime.now().strftime(
    #     "%Y-%m-%d_%H%M%S.%f"
    # )
    # # Create dirs that do not exist
    # args.save_dir.mkdir(parents=True, exist_ok=True)
    # arg_parser.save_namespace(args, str(args.save_dir / "params.pkl"))
    # args.save_dir = str(args.save_dir)
    print(args)

if __name__ == "__main__":
    main()