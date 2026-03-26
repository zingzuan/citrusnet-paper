from citrusnet.models import build_model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    configs = [
        "configs/model/mobilevit.yaml",
        "configs/model/citrusnet_v1.yaml",
        "configs/model/citrusnet_v2.yaml",
        "configs/model/citrusnet_v3.yaml",
    ]

    for cfg in configs:
        model = build_model(cfg)
        params = count_params(model)
        print(f"{cfg}: {params / 1e6:.4f} M")


if __name__ == "__main__":
    main()