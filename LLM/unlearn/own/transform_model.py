import re

import torch
import torch.nn as nn
from torch import linalg as LA
from torch.utils.data import DataLoader

from .utils import replace_layers_with_custom


def transform_text_layer(text):
    return re.sub(r"(^|\.)([0-9]+)(?=\.|$)", lambda match: f"[{match.group(2)}]", text)

def get_lora_module_from_param_name(model: nn.Module, param_name: str):
    """
    Given a LoRA parameter name, return the parent module
    that contains lora_A and lora_B.
    """
    # usuwamy końcówkę '.lora_A.weight' albo '.lora_B.weight'
    if ".lora_A." in param_name:
        module_path = param_name.split(".lora_A.")[0]
    elif ".lora_B." in param_name:
        module_path = param_name.split(".lora_B.")[0]
    else:
        raise ValueError(f"Not a LoRA parameter: {param_name}")

    module = model
    for attr in module_path.split("."):
        module = getattr(module, attr)

    return module

def get_lora_delta_weight(lora_module: nn.Module):
    """
    Compute ΔW = B @ A for a LoRA module.
    """
    adapter_name = "default"
    A_layer = lora_module.lora_A[adapter_name]   
    B_layer = lora_module.lora_B[adapter_name]  

    A = A_layer.weight
    B = B_layer.weight

    delta_W =  B @ A
    delta_W = delta_W.to("cpu")     

    return delta_W           

def transform_model(
    model: nn.Module,
    data_loader_unlearn: DataLoader,
    criterion: nn.Module,
    explained_variance_ratio: float = None,
    use_projection_grad: bool = False,
) -> None:
    
    print("Transforming model")

    def compute_gradients(batch):

        loss = criterion(model, batch)
        loss.backward()

        gradients_dict = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients_dict[name[: name.rfind(".")]] = param.grad.clone()
        return gradients_dict, loss.item()
    
   
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    
    print("Computing gradients")

    sum_gradients = None

    loader_len = len(data_loader_unlearn)
    for i, batch in enumerate(data_loader_unlearn):
        grads, _ = compute_gradients(batch)

        if sum_gradients is None:
            sum_gradients = grads
        else:
            for key, val in grads.items():
                sum_gradients[key] += val

        del grads
        torch.cuda.empty_cache()

    print(f"### USE_PROJECTION_GRAD: {use_projection_grad} ###")
    if use_projection_grad:
        for param_name, grad in sum_gradients.items():
            param = eval(f"model.{transform_text_layer(param_name)}.weight")

            grad_flat = grad.view(-1)
            param_flat = param.view(-1)
            proj_grad_flat = grad_flat - (
                torch.dot(grad_flat, param_flat)
                / (torch.norm(param_flat) ** 2 + 1e-12)
            ) * param_flat
            sum_gradients[param_name] = proj_grad_flat.view_as(grad)

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

    print("Replacing layers")

    replace_layers_with_custom(model, u_matrices, vh_matrices)
