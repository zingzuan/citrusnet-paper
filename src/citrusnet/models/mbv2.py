import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MbV2Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        backbone = mobilenet_v2(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)