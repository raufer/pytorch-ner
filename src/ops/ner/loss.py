import torch

from typing import Callable
from torch import Tensor


def active_loss(loss_fct: Callable, target: Tensor, logits: Tensor, target_ignore_index=-1) -> Tensor:
    """
    Returns the active part of the loss
    "active" is defined by the entries in the `target` tensor with values != `target_ignore_index`
    """
    active_loss = target.view(-1) != target_ignore_index
    num_labels = logits.shape[-1]

    active_logits = logits.view(-1, num_labels)
    active_labels = torch.where(active_loss, target.view(-1), torch.tensor(loss_fct.ignore_index).type_as(target))

    loss = loss_fct(active_logits, active_labels)
    return loss
