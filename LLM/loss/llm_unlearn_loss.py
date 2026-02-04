from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, batch):
        """
        model  : trainable LLM
        batch : input batch
        """

        model.zero_grad()

        device = model.get_input_embeddings().weight.device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, return_dict=True)
        loss = outputs.loss

        return loss
