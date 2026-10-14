
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
WEIGHTS_PATH = (
    ROOT
    / "runs"
    / "detection"
    / "yolov8n_minecraft_mobs"
    / "weights"
    / "best.pt"
)
REPORT_DIR = ROOT / "reports" / "detection"
REPORT_PATH = REPORT_DIR / "evaluation_report.json"


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_root = ROOT / config["dataset"]["root"]
    data_yaml = dataset_root / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    expected_names = config["dataset"]["classes"]
    dataset_names = dataset_config["names"]

    if isinstance(dataset_names, dict):
        dataset_names = [
            dataset_names[index]
            for index in range(len(dataset_names))
        ]

    if list(dataset_names) != list(expected_names):
        raise ValueError(
            "Class order mismatch!\n"
            f"Project config: {expected_names}\n"
            f"Dataset config: {dataset_names}"
        )

    print(f"Model: {WEIGHTS_PATH}")
    print(f"Dataset: {data_yaml}")
    print("Split: validation")
    print(f"Classes: {expected_names}")
    print("\nRunning validation...\n")

    model = YOLO(str(WEIGHTS_PATH))

    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=config["detection"]["image_size"],
        batch=config["detection"]["batch_size"],
        workers=config["detection"]["workers"],
        device=0 if config["hardware"]["device"] == "cuda" else "cpu",
        plots=False,
        verbose=True,
    )

    box_metrics = metrics.box

    per_class = {}
    for class_id, class_name in enumerate(expected_names):
        per_class[class_name] = {
            "precision": float(box_metrics.p[class_id]),
            "recall": float(box_metrics.r[class_id]),
            "map50": float(box_metrics.ap50[class_id]),
            "map50_95": float(box_metrics.ap[class_id].mean()),
        }

    report = {
        "project": "SceneForgeCV",
        "task": "object_detection",
        "model": "YOLOv8n",
        "checkpoint": str(WEIGHTS_PATH.relative_to(ROOT)),
        "dataset": str(data_yaml.relative_to(ROOT)),
        "split": "val",
        "image_size": config["detection"]["image_size"],
        "metrics": {
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
            "map50": float(box_metrics.map50),
            "map50_95": float(box_metrics.map),
        },
        "per_class": per_class,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 55)
    print("OVERALL VALIDATION METRICS")
    print("=" * 55)
    for name, value in report["metrics"].items():
        print(f"{name:12s}: {value:.4f}")

    print("\nPER-CLASS METRICS")
    print("-" * 75)
    print(
        f"{'Class':12s} {'Precision':>10s} {'Recall':>10s} "
        f"{'mAP50':>10s} {'mAP50-95':>10s}"
    )

    for class_name, values in per_class.items():
        print(
            f"{class_name:12s} "
            f"{values['precision']:10.4f} "
            f"{values['recall']:10.4f} "
            f"{values['map50']:10.4f} "
            f"{values['map50_95']:10.4f}"
        )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
