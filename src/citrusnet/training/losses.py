import torch.nn as nn


def build_criterion(name: str):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported criterion: {name}")