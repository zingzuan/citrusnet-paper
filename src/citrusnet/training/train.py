import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from citrusnet.data.datasets import build_dataloaders
from citrusnet.models import build_model
from citrusnet.training.evaluate import evaluate_model
from citrusnet.training.losses import build_criterion
from citrusnet.training.metrics import save_confusion_matrix


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else 0.0


def main(config_path: str):
    train_cfg = load_yaml(config_path)
    data_cfg = load_yaml("configs/data.yaml")
    model_cfg_path = train_cfg["model_config"]
    model_cfg = load_yaml(model_cfg_path)

    device = resolve_device(train_cfg["device"])

    train_loader, val_loader, test_loader = build_dataloaders(
        data_cfg_path="configs/data.yaml",
        train_cfg_path=config_path,
    )

    model = build_model(model_cfg_path).to(device)
    criterion = build_criterion(train_cfg["criterion"])

    if train_cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=train_cfg["lr_decay_factor"],
    )

    checkpoint_dir = Path(train_cfg["checkpoint_dir"]) / model_cfg["name"]
    metrics_dir = Path(train_cfg["metrics_dir"]) / model_cfg["name"]
    plots_dir = Path(train_cfg["plots_dir"]) / model_cfg["name"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate_model(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

        scheduler.step()

    torch.save(model.state_dict(), checkpoint_dir / "last.pt")

    test_metrics, y_true, y_pred = evaluate_model(model, test_loader, device)

    with open(metrics_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(metrics_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=data_cfg["class_names"],
        output_path=str(plots_dir / "confusion_matrix.png"),
    )

    print("test_metrics:", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)