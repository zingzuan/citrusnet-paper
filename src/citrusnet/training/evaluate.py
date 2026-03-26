import torch

from citrusnet.training.metrics import compute_classification_metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total_samples if total_samples > 0 else 0.0

    return metrics, all_labels, all_preds