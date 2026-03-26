from citrusnet.data.datasets import build_dataloaders


def test_dataloaders_build():
    train_loader, val_loader, test_loader = build_dataloaders()

    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0

    images, labels = next(iter(train_loader))

    assert images.ndim == 4
    assert images.shape[1] == 3
    assert images.shape[2] == 256
    assert images.shape[3] == 256
    assert labels.ndim == 1