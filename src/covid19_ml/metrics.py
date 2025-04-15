import torch
from torch import Tensor


def nse_loss(input: Tensor, target: Tensor) -> Tensor:
    return 1 - torch.sum((input - target) ** 2) / torch.sum(
        (target - torch.mean(target)) ** 2
    )


def log_rmse_loss(y_true, y_pred):
    return torch.sqrt(torch.mean((torch.log1p(y_true) - torch.log1p(y_pred)) ** 2))
