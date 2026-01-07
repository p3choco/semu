from torch.utils.data import Dataset
from datasets import load_dataset

class BLUR(Dataset):
    def __init__(self, hf_dataset, task, variant):
        super().__init__()
        self.data = hf_dataset["train"]
        self.task = task
        self.variant = variant

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.data[idx]["text"],
            "task": self.task,
            "variant": self.variant
        }
    
def get_BLUR_dataset(task, variant):
    """
    task: 'rwku' | 'whp' | 'tofu' | 'wmdp'
    variant: 'forget' | 'retain' | 'paired_forget_retain' | 'D_hi' | 'D_mid' | 'D_low'
    """

    valid_tasks = ["rwku", "whp", "tofu", "wmdp"]
    valid_variants = ["forget", "retain", "paired_forget_retain", "D_hi", "D_mid", "D_lo"]

    if task not in valid_tasks:
        raise ValueError(f"Invalid task '{task}'. Valid tasks are: {valid_tasks}")
    if variant not in valid_variants:
        raise ValueError(f"Invalid variant '{variant}'. Valid variants are: {valid_variants}")

    if task=="tofu" and variant!="paired_forget_retain":
        raise ValueError("For TOFU task, only 'paired_forget_retain' variant is available.")
    if task=="wmdp" and variant=="forget":
        raise ValueError("For WMDP task, 'forget' variant in currently unavailable.")

    BLUR_REPO_ID = "forgelab/BLUR"
    if variant=="D_lo":
        filename = "D_lo_for_all.json"
    else:
        filename = f"{task}_{variant}.json"

    hf_dataset = load_dataset(
        BLUR_REPO_ID,
        data_files=filename)
    
    return BLUR(hf_dataset, task, variant)