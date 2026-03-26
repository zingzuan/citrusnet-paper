import random
import shutil
from pathlib import Path

import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(files, output_dir: Path, class_name: str):
    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, class_dir / file_path.name)


def split_files(files, train_ratio, val_ratio, test_ratio, seed):
    files = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:n_train + n_val + n_test]

    return train_files, val_files, test_files


def main():
    cfg = load_yaml("configs/data.yaml")

    raw_citrus_dir = Path(cfg["raw_citrus_dir"])
    train_dir = Path(cfg["train_dir"])
    val_dir = Path(cfg["val_dir"])
    test_dir = Path(cfg["test_dir"])

    split_cfg = cfg["split"]
    seed = cfg["seed"]
    class_names = cfg["class_names"]

    if not raw_citrus_dir.exists():
        raise FileNotFoundError(f"Raw citrus directory not found: {raw_citrus_dir}")

    clear_dir(train_dir)
    clear_dir(val_dir)
    clear_dir(test_dir)

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name in class_names:
        class_input_dir = raw_citrus_dir / class_name
        if not class_input_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_input_dir}")

        image_files = [
            p for p in class_input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ]

        if len(image_files) == 0:
            raise ValueError(f"No images found in {class_input_dir}")

        train_files, val_files, test_files = split_files(
            image_files,
            split_cfg["train"],
            split_cfg["val"],
            split_cfg["test"],
            seed,
        )

        copy_split(train_files, train_dir, class_name)
        copy_split(val_files, val_dir, class_name)
        copy_split(test_files, test_dir, class_name)

        print(
            f"{class_name}: total={len(image_files)} "
            f"train={len(train_files)} val={len(val_files)} test={len(test_files)}"
        )

    print("Finished preparing data.")


if __name__ == "__main__":
    main()