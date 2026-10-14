
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_root = (ROOT / config["dataset"]["root"]).resolve()
    data_yaml = dataset_root / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file)

    expected_classes = config["dataset"]["classes"]
    actual_classes = dataset_config["names"]

    if isinstance(actual_classes, dict):
        actual_classes = [
            actual_classes[key]
            for key in sorted(actual_classes, key=lambda key: int(key))
        ]

    if actual_classes != expected_classes:
        raise ValueError(
            f"Class order mismatch. Expected {expected_classes}, "
            f"got {actual_classes}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'cuda:0' if device == 0 else 'cpu'}")

    if device == 0:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = config["detection"]["model"]
    model = YOLO(model_name)

    output_dir = ROOT / "runs" / "detection"

    results = model.train(
        data=str(data_yaml),
        imgsz=config["detection"]["image_size"],
        batch=config["detection"]["batch_size"],
        epochs=config["detection"]["epochs"],
        patience=config["detection"]["patience"],
        workers=config["detection"]["workers"],
        seed=config["detection"]["seed"],
        device=device,
        project=str(output_dir),
        name="yolov8n_minecraft_mobs",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        amp=True,
        cache=False,
        plots=False,
        save=True,
        save_period=5,
        val=True,
        verbose=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    metrics = getattr(model.trainer, "metrics", None)
    metrics_dict = getattr(metrics, "results_dict", {}) or {}

    serializable_metrics = {
        str(key): float(value)
        for key, value in metrics_dict.items()
        if isinstance(value, (int, float))
    }

    report = {
        "model": model_name,
        "dataset": str(data_yaml),
        "classes": expected_classes,
        "epochs_configured": config["detection"]["epochs"],
        "run_directory": str(run_dir),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "metrics": serializable_metrics,
    }

    report_dir = ROOT / "reports" / "detection"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "training_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nDetection training finished.")
    print(f"Run directory: {run_dir}")
    print(f"Best weights: {best_weights}")
    print(f"Training summary: {report_path}")
    print("\nMetrics:")
    for key, value in serializable_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
