import json
import subprocess
from pathlib import Path

import yaml


CONFIG_PATH = "configs/train.yaml"
MODEL_CONFIGS = [
    "configs/model/mbv2.yaml",
    "configs/model/mobilevit.yaml",
    "configs/model/citrusnet_v1.yaml",
    "configs/model/citrusnet_v2.yaml",
    "configs/model/citrusnet_v3.yaml",
]


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    base_cfg = load_yaml(CONFIG_PATH)

    for model_cfg in MODEL_CONFIGS:
        cfg = dict(base_cfg)
        cfg["model_config"] = model_cfg
        save_yaml(CONFIG_PATH, cfg)

        print(f"\n=== Running {model_cfg} ===")
        result = subprocess.run(
            ["python", "-m", "citrusnet.training.train", "--config", CONFIG_PATH],
            check=False,
        )

        if result.returncode != 0:
            print(f"Run failed for {model_cfg}")
            break

    save_yaml(CONFIG_PATH, base_cfg)
    print("\nRestored original train config.")


if __name__ == "__main__":
    main()