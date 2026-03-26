import json
from pathlib import Path


MODEL_NAMES = [
    "mbv2",
    "mobilevit",
    "citrusnet_v1",
    "citrusnet_v2",
    "citrusnet_v3",
]


def main():
    rows = []

    for name in MODEL_NAMES:
        metrics_path = Path("artifacts/metrics") / name / "test_metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        row = {
            "model": name,
            "accuracy": metrics.get("accuracy"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "f1_macro": metrics.get("f1_macro"),
            "loss": metrics.get("loss"),
        }
        rows.append(row)

    output_path = Path("artifacts") / "backbone_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()