from torch import nn

from ..torch_utility import Swish

__all__ = ["create_activation"]


def create_activation(activation_type: str) -> nn.Module:
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "swish":
        return Swish()
    elif activation_type == "none":
        return nn.Identity()
    raise ValueError("invalid activation_type.")
