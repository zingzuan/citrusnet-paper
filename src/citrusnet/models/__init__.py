import yaml

from citrusnet.models.citrusnet import CitrusNet
from citrusnet.models.mbv2 import MbV2Classifier
from citrusnet.models.mobilevit import MobileViT


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_cfg_path: str):
    cfg = load_yaml(model_cfg_path)
    name = cfg["name"]
    num_classes = cfg.get("num_classes", 5)
    dropout = cfg.get("dropout", 0.1)

    if name == "mbv2":
        return MbV2Classifier(num_classes=num_classes, dropout=dropout)

    if name == "mobilevit":
        return MobileViT(
            num_classes=num_classes,
            dropout=dropout,
            mv2_expand_ratio=cfg.get("mv2_expand_ratio", 4),
            mobilevit_dims=tuple(cfg.get("mobilevit_dims", [96, 120, 144])),
            mobilevit_depths=tuple(cfg.get("mobilevit_depths", [2, 4, 3])),
            patch_size=tuple(cfg.get("patch_size", [2, 2])),
        )

    if name in {"citrusnet_v1", "citrusnet_v2", "citrusnet_v3"}:
        return CitrusNet(
            num_classes=num_classes,
            dropout=dropout,
            variant=cfg["variant"],
            mv2_expand_ratio=cfg.get("mv2_expand_ratio", 4),
            mobilevit_dims=tuple(cfg.get("mobilevit_dims", [96, 120, 144])),
            mobilevit_depths=tuple(cfg.get("mobilevit_depths", [2, 4, 3])),
            patch_size=tuple(cfg.get("patch_size", [2, 2])),
        )

    raise ValueError(f"Unknown model name: {name}")