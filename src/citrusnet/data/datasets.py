from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import datasets

from citrusnet.data.transforms import (
    build_eval_transforms,
    build_ssl_transforms,
    build_train_transforms,
)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_datasets(data_cfg_path: str = "configs/data.yaml"):
    data_cfg = load_yaml(data_cfg_path)
    image_size = data_cfg["image_size"]

    train_ds = datasets.ImageFolder(
        data_cfg["train_dir"],
        transform=build_train_transforms(image_size),
    )
    val_ds = datasets.ImageFolder(
        data_cfg["val_dir"],
        transform=build_eval_transforms(image_size),
    )
    test_ds = datasets.ImageFolder(
        data_cfg["test_dir"],
        transform=build_eval_transforms(image_size),
    )

    return train_ds, val_ds, test_ds


def build_dataloaders(
    data_cfg_path: str = "configs/data.yaml",
    train_cfg_path: str = "configs/train.yaml",
):
    train_cfg = load_yaml(train_cfg_path)

    train_ds, val_ds, test_ds = build_datasets(data_cfg_path)

    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    pin_memory = train_cfg["pin_memory"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def build_ssl_dataset(data_cfg_path: str = "configs/data.yaml"):
    data_cfg = load_yaml(data_cfg_path)
    image_size = data_cfg["image_size"]
    ssl_root = data_cfg["raw_plantvillage_dir"]

    if not Path(ssl_root).exists():
        raise FileNotFoundError(f"SSL dataset directory not found: {ssl_root}")

    return datasets.ImageFolder(
        ssl_root,
        transform=build_ssl_transforms(image_size),
    )