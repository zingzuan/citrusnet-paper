import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=not use_norm,
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MV2Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int = 4):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                )
            )

        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int,
        patch_h: int = 2,
        patch_w: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.local_rep = nn.Sequential(
            ConvLayer(in_channels, in_channels, kernel_size=3, stride=1),
            ConvLayer(in_channels, transformer_dim, kernel_size=1, stride=1),
        )

        self.transformer = nn.Sequential(
            *[
                TransformerEncoder(
                    dim=transformer_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                )
                for _ in range(n_transformer_blocks)
            ]
        )

        self.project = ConvLayer(transformer_dim, in_channels, kernel_size=1, stride=1)
        self.fusion = ConvLayer(in_channels * 2, in_channels, kernel_size=3, stride=1)

    def unfolding(self, x: torch.Tensor):
        b, c, h, w = x.shape
        new_h = int(math.ceil(h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(w / self.patch_w) * self.patch_w)

        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        num_patch_h = new_h // self.patch_h
        num_patch_w = new_w // self.patch_w
        num_patches = num_patch_h * num_patch_w
        patch_area = self.patch_h * self.patch_w

        x = x.reshape(b, c, num_patch_h, self.patch_h, num_patch_w, self.patch_w)
        x = x.permute(0, 3, 5, 2, 4, 1)
        x = x.reshape(b * patch_area, num_patches, c)

        return x, (h, w), (new_h, new_w)

    def folding(self, x: torch.Tensor, orig_size, new_size):
        orig_h, orig_w = orig_size
        new_h, new_w = new_size
        num_patch_h = new_h // self.patch_h
        num_patch_w = new_w // self.patch_w
        patch_area = self.patch_h * self.patch_w
        b = x.shape[0] // patch_area
        c = x.shape[-1]

        x = x.reshape(b, self.patch_h, self.patch_w, num_patch_h, num_patch_w, c)
        x = x.permute(0, 5, 3, 1, 4, 2)
        x = x.reshape(b, c, new_h, new_w)

        if new_h != orig_h or new_w != orig_w:
            x = F.interpolate(x, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.local_rep(x)
        x, orig_size, new_size = self.unfolding(x)
        x = self.transformer(x)
        x = self.folding(x, orig_size, new_size)
        x = self.project(x)
        x = torch.cat([residual, x], dim=1)
        x = self.fusion(x)
        return x