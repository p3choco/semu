from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMQuestionOnlyUnlearningLoss(nn.Module):
    """
    Question-only unlearning loss using a frozen reference model.

    Loss =
        - KL( p_ref(.|q) || p_model(.|q) )

    This forgets question-level behavior without requiring answers.
    """

    def __init__(
        self,
        ref_model=None,
        temperature=2.0,
        device=None,
    ):
        super().__init__()

        # self.ref_model = ref_model
        # self.temperature = temperature
        # self.device = device or next(ref_model.parameters()).device

        # # Freeze reference model
        # self.ref_model.eval()
        # for p in self.ref_model.parameters():
        #     p.requires_grad = False

    def forward(self, model, batch):
        """
        model  : trainable LLM
        prompt : str
        """

        model.zero_grad()
        # Current model distribution
        device = model.get_input_embeddings().weight.device

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, return_dict=True)
        loss = outputs.loss

        return loss, {
            # "forget_kl": loss.detach(),
        }
