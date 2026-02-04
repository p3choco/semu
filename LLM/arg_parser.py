import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="LLM unlearning")
    parser.add_argument("--save_dir", type=str, default="UL")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--unlearn", type=str, default="own_SVD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="rwku")
    parser.add_argument("--decreasing_lr", type=str, default="30")
    parser.add_argument("--unlearn_lr", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--unlearn_epochs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=10)
    return parser.parse_args()
