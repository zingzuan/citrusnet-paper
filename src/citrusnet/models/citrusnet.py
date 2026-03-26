import torch
import torch.nn as nn

from citrusnet.models.mobilevit_blocks import ConvLayer, MobileViTBlock
from citrusnet.models.mbv2se import MbV2SEBlock


class CitrusNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.1,
        variant: str = "v2",
        mv2_expand_ratio: int = 4,
        mobilevit_dims=(96, 120, 144),
        mobilevit_depths=(2, 4, 3),
        patch_size=(2, 2),
    ):
        super().__init__()

        if variant not in {"v1", "v2", "v3"}:
            raise ValueError(f"Unsupported CitrusNet variant: {variant}")

        d1, d2, d3 = mobilevit_dims
        l1, l2, l3 = mobilevit_depths
        ph, pw = patch_size

        repeat_4th_layer = {
            "v1": 2,
            "v2": 1,
            "v3": 0,
        }[variant]

        self.conv_1 = ConvLayer(3, 16, kernel_size=3, stride=2)

        self.layer_1 = MbV2SEBlock(16, 32, stride=1, expand_ratio=mv2_expand_ratio)
        self.layer_2 = MbV2SEBlock(32, 48, stride=2, expand_ratio=mv2_expand_ratio)

        layer_3_blocks = []
        for _ in range(repeat_4th_layer):
            layer_3_blocks.append(MbV2SEBlock(48, 48, stride=1, expand_ratio=mv2_expand_ratio))
        self.layer_3 = nn.Sequential(*layer_3_blocks) if layer_3_blocks else nn.Identity()

        self.layer_4 = nn.Sequential(
            MbV2SEBlock(48, 64, stride=2, expand_ratio=mv2_expand_ratio),
            MobileViTBlock(
                in_channels=64,
                transformer_dim=d1,
                ffn_dim=d1 * 2,
                n_transformer_blocks=l1,
                patch_h=ph,
                patch_w=pw,
            ),
        )

        self.layer_5 = nn.Sequential(
            MbV2SEBlock(64, 80, stride=2, expand_ratio=mv2_expand_ratio),
            MobileViTBlock(
                in_channels=80,
                transformer_dim=d2,
                ffn_dim=d2 * 2,
                n_transformer_blocks=l2,
                patch_h=ph,
                patch_w=pw,
            ),
        )

        self.layer_6 = nn.Sequential(
            MbV2SEBlock(80, 96, stride=2, expand_ratio=mv2_expand_ratio),
            MobileViTBlock(
                in_channels=96,
                transformer_dim=d3,
                ffn_dim=d3 * 2,
                n_transformer_blocks=l3,
                patch_h=ph,
                patch_w=pw,
            ),
        )

        self.conv_1x1_exp = ConvLayer(96, 384, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.conv_1x1_exp(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x