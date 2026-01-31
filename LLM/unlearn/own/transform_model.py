import re

import torch
import torch.nn as nn
from torch import linalg as LA
from torch.utils.data import DataLoader

from .utils import set_requires_grad, replace_layers_with_custom


def transform_text_layer(text):
    return re.sub(r"(^|\.)([0-9]+)(?=\.|$)", lambda match: f"[{match.group(2)}]", text)


def transform_model(
    model: nn.Module,
    data_loader_unlearn: DataLoader,
    criterion: nn.Module,
    changed_layers_class: list[str] = None,
    explained_variance_ratio: float = None,
    use_projection_grad: bool = False,
) -> None:
    """
    Transform the model by replacing the last layer with a new linear layer.

    Args:
        model (nn.Module): The input model.
        data_loader_unlearn (DataLoader): DataLoader for the unlearning dataset.
        criterion (nn.Module): Loss function.
        changed_layers_class (list[str], optional): List of layer class names for which parameters should remain trainable.
        explained_variance_ratio (float, optional): Explained variance ratio for the new linear layer.
        use_projection_grad: Make projection gradients onto the space perpendicular to the weights

    Returns:
        nn.Module: The transformed model.
    """
    print("Transforming model")
    device = next(model.parameters()).device

    if changed_layers_class is None:
        changed_layers_class = ["linear", "conv2d"]

    def compute_gradients(batch):
        """
        Compute gradients for LLM unlearning using blur (question-only) data
        via negative KL divergence from a reference model.
        """

        loss, metrics = criterion(model, batch)
        # Backprop
        loss.backward()

        gradients_dict = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # move gradient to CPU and detach graph
                gradients_dict[name] = param.grad.detach().cpu()

        return gradients_dict, loss.item()

    set_requires_grad(model, changed_layers_class=changed_layers_class)
    print("Compute gradients")
    sum_gradients = None
    for i, batch in enumerate(data_loader_unlearn):
        print(f"Batch {i}")

        grads, loss_val = compute_gradients(batch)

        if sum_gradients is None:
            sum_gradients = grads
        else:
            for k in sum_gradients:
                sum_gradients[k] += grads[k]

        # explicit cleanup
        del grads
        torch.cuda.empty_cache()

    if use_projection_grad:
        # Projecting the gradient onto the space perpendicular to the weights
        for layer_name, grad in sum_gradients.items():
            _weight = eval(f"model.{transform_text_layer(layer_name)}.weight")
            proj_grad = grad - (torch.sum(grad * _weight) / LA.norm(_weight)) * _weight
            sum_gradients[layer_name] = proj_grad

    u_matrices = {}
    vh_matrices = {}
    print("SVD computing")
    for key, G in sum_gradients.items():
        if G.dim() == 2:
            u, s, vh = torch.linalg.svd(G, full_matrices=False)
            if explained_variance_ratio is None:
                u_, vh_ = u, vh
            else:
                singular_values_squared = torch.square(s)
                total_variance = singular_values_squared.sum()
                cumulative_variance = torch.cumsum(singular_values_squared, dim=0)
                explained_variance = cumulative_variance / total_variance
                k = torch.searchsorted(explained_variance, explained_variance_ratio, side="right")
                k = max(1, k)
                # k = max(1, int(s.shape[0] * explained_variance_ratio))

                u_ = torch.empty((u.shape[0], k), device=u.device, dtype=u.dtype)
                vh_ = torch.empty((k, vh.shape[1]), device=vh.device, dtype=vh.dtype)
                u_.data.copy_(u[:, :k])
                vh_.data.copy_(vh[:k, :])
        elif G.dim() == 4:
            _weight = torch.permute(G, (1, 0, 2, 3))
            _weight = torch.flatten(_weight, start_dim=2, end_dim=-1)

            u, s, vh = torch.linalg.svd(_weight, full_matrices=False)

            if explained_variance_ratio is None:
                u_, vh_ = u, vh
            else:
                singular_values_squared = torch.square(s)
                total_variance = singular_values_squared.sum(dim=1, keepdim=True)
                cumulative_variance = torch.cumsum(singular_values_squared, dim=1)
                explained_variance = cumulative_variance / total_variance
                k = torch.searchsorted(explained_variance,
                                       torch.full((s.shape[0], 1), explained_variance_ratio, device=s.device),
                                       side="right")
                k = max(1, k.max())
                # k = max(1, int(s.shape[1] * explained_variance_ratio))

                u_ = torch.empty((u.shape[0], u.shape[1], k), device=u.device, dtype=u.dtype)
                vh_ = torch.empty((vh.shape[0], k, vh.shape[2]), device=vh.device, dtype=vh.dtype)
                u_.data.copy_(u[:, :, :k])
                vh_.data.copy_(vh[:, :k, :])
        else:
            raise NotImplemented(
                "Operations are only suitable for layers: Conv2d and Linear"
            )
        u_matrices[key] = u_
        vh_matrices[key] = vh_

    print("Replace layers")

    replace_layers_with_custom(model, u_matrices, vh_matrices)
