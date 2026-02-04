
from unlearn import load_unlearn_checkpoint
import torch 
def main():
    # DISCLAIMER - if model needs to be loaded, CustomLinear should be in the same place as during saving. 
    model = torch.load("UL/unlearned_model.llama", weights_only=False)
    for name, p in model.named_modules():
        print(name, type(p))
    
if __name__ == "__main__":
    main()