import torch
import torch.nn as nn

from citrusnet.models.mobilevit_blocks import ConvLayer, MV2Block, MobileViTBlock


class MobileViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.1,
        mv2_expand_ratio: int = 4,
        mobilevit_dims=(96, 120, 144),
        mobilevit_depths=(2, 4, 3),
        patch_size=(2, 2),
    ):
        super().__init__()

        d1, d2, d3 = mobilevit_dims
        l1, l2, l3 = mobilevit_depths
        ph, pw = patch_size

        self.conv_1 = ConvLayer(3, 16, kernel_size=3, stride=2)

        self.layer_1 = MV2Block(16, 32, stride=1, expand_ratio=mv2_expand_ratio)
        self.layer_2 = MV2Block(32, 48, stride=2, expand_ratio=mv2_expand_ratio)

        self.layer_3 = nn.Sequential(
            MV2Block(48, 48, stride=1, expand_ratio=mv2_expand_ratio),
            MV2Block(48, 48, stride=1, expand_ratio=mv2_expand_ratio),
        )

        self.layer_4 = nn.Sequential(
            MV2Block(48, 64, stride=2, expand_ratio=mv2_expand_ratio),
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
            MV2Block(64, 80, stride=2, expand_ratio=mv2_expand_ratio),
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
            MV2Block(80, 96, stride=2, expand_ratio=mv2_expand_ratio),
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