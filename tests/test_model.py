import torch

from citrusnet.models import build_model


def _check_forward(model_cfg_path: str):
    model = build_model(model_cfg_path)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 5)


def test_mbv2_forward():
    _check_forward("configs/model/mbv2.yaml")


def test_mobilevit_forward():
    _check_forward("configs/model/mobilevit.yaml")


def test_citrusnet_v1_forward():
    _check_forward("configs/model/citrusnet_v1.yaml")


def test_citrusnet_v2_forward():
    _check_forward("configs/model/citrusnet_v2.yaml")


def test_citrusnet_v3_forward():
    _check_forward("configs/model/citrusnet_v3.yaml")