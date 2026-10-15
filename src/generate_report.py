
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
REPORT_DIR = ROOT / "reports"
JSON_PATH = REPORT_DIR / "project_summary.json"
MARKDOWN_PATH = REPORT_DIR / "project_summary.md"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSIFIER_CHECKPOINT = (
    ROOT / "checkpoints" / "classification" / "resnet18_best.pth"
)
DETECTOR_CHECKPOINT = (
    ROOT
    / "runs"
    / "detection"
    / "yolov8n_minecraft_mobs"
    / "weights"
    / "best.pt"
)


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def audit_detection_split(
    dataset_root: Path,
    split: str,
    class_names: list[str],
) -> dict:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    image_count = count_images(image_dir)
    label_files = (
        list(label_dir.glob("*.txt"))
        if label_dir.is_dir()
        else []
    )

    class_counts = Counter()
    empty_label_files = 0
    invalid_lines = 0
    total_boxes = 0

    for label_file in label_files:
        try:
            lines = [
                line.strip()
                for line in label_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
        except OSError:
            invalid_lines += 1
            continue

        if not lines:
            empty_label_files += 1
            continue

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                invalid_lines += 1
                continue

            if (
                class_id < 0
                or class_id >= len(class_names)
                or any(value < 0 or value > 1 for value in coordinates)
            ):
                invalid_lines += 1
                continue

            if coordinates[2] <= 0 or coordinates[3] <= 0:
                invalid_lines += 1
                continue

            class_counts[class_names[class_id]] += 1
            total_boxes += 1

    return {
        "images": image_count,
        "label_files": len(label_files),
        "empty_label_files": empty_label_files,
        "bounding_box_instances": total_boxes,
        "instances_per_class": {
            name: class_counts[name] for name in class_names
        },
        "invalid_annotation_lines": invalid_lines,
    }


def count_classification_crops(
    directory: Path,
    class_names: list[str],
) -> dict:
    counts = {}

    for class_name in class_names:
        class_dir = directory / class_name
        counts[class_name] = count_images(class_dir)

    return {
        "total": sum(counts.values()),
        "per_class": counts,
    }


def artifact_status(path: Path) -> dict:
    exists = path.is_file()

    result = {
        "exists": exists,
        "path": str(path.relative_to(ROOT)),
    }

    if exists:
        result["size_bytes"] = path.stat().st_size

    return result


def build_markdown(summary: dict) -> str:
    lines = [
        "# SceneForgeCV — Project Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Dataset",
        "",
        f"- Dataset root: `{summary['dataset']['root']}`",
        f"- Classes: {', '.join(summary['dataset']['classes'])}",
        "",
        "| Split | Images | Label files | Bounding boxes | Empty labels | Invalid lines |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for split_name, data in summary["dataset"]["detection"].items():
        lines.append(
            f"| {split_name} | {data['images']} | "
            f"{data['label_files']} | "
            f"{data['bounding_box_instances']} | "
            f"{data['empty_label_files']} | "
            f"{data['invalid_annotation_lines']} |"
        )

    lines.extend(
        [
            "",
            "### Detection instances per class",
            "",
            "| Class | Train | Validation |",
            "|---|---:|---:|",
        ]
    )

    train_counts = summary["dataset"]["detection"]["train"][
        "instances_per_class"
    ]
    val_counts = summary["dataset"]["detection"]["val"][
        "instances_per_class"
    ]

    for class_name in summary["dataset"]["classes"]:
        lines.append(
            f"| {class_name} | {train_counts[class_name]} | "
            f"{val_counts[class_name]} |"
        )

    lines.extend(
        [
            "",
            "### Classification crops",
            "",
            "| Split | Total crops | Per-class counts |",
            "|---|---:|---|",
        ]
    )

    for split_name, data in summary["dataset"]["classification"].items():
        class_text = ", ".join(
            f"{name}: {count}"
            for name, count in data["per_class"].items()
        )
        lines.append(f"| {split_name} | {data['total']} | {class_text} |")

    lines.extend(
        [
            "",
            "## Model configurations",
            "",
            "| Model | Image size | Batch size | Configured epochs |",
            "|---|---:|---:|---:|",
        ]
    )

    for model_name, data in summary["model_configuration"].items():
        lines.append(
            f"| {model_name} | {data['image_size']} | "
            f"{data['batch_size']} | {data['epochs']} |"
        )

    lines.extend(
        [
            "",
            "## Model artifacts and reports",
            "",
            "| Artifact | Available | Path |",
            "|---|---|---|",
        ]
    )

    for name, data in summary["artifacts"].items():
        status = "Yes" if data["exists"] else "No"
        lines.append(f"| {name} | {status} | `{data['path']}` |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Dataset counts are discovered from files on disk.",
            "- Configured epochs are not necessarily epochs completed.",
            "- Checkpoint existence does not establish model quality.",
            "- See evaluation reports for measured model performance.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    config = load_yaml(CONFIG_PATH)

    dataset_root = ROOT / config["dataset"]["root"]
    data_yaml_path = dataset_root / "data.yaml"
    data_config = load_yaml(data_yaml_path)

    class_names = config["dataset"]["classes"]
    configured_names = data_config["names"]

    if isinstance(configured_names, dict):
        configured_names = [
            configured_names[index]
            for index in range(len(configured_names))
        ]

    if list(configured_names) != list(class_names):
        raise ValueError(
            "Class order mismatch between configs/config.yaml and "
            "dataset data.yaml."
        )

    detection_splits = {
        split: audit_detection_split(
            dataset_root,
            split,
            class_names,
        )
        for split in ("train", "val")
    }

    processed_root = (
        ROOT
        / config["dataset"]["processed_root"]
        / "classification"
    )

    classification_splits = {
        split: count_classification_crops(
            processed_root / split,
            class_names,
        )
        for split in ("train", "val")
    }

    detection_config = config["detection"]
    classification_config = config["classification"]

    summary = {
        "project": config["project"]["name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "root": str(dataset_root.relative_to(ROOT)),
            "classes": class_names,
            "num_classes": len(class_names),
            "detection": detection_splits,
            "classification": classification_splits,
        },
        "model_configuration": {
            "ResNet18": {
                "image_size": classification_config["image_size"],
                "batch_size": classification_config["batch_size"],
                "epochs": classification_config["epochs"],
                "learning_rate": classification_config["learning_rate"],
            },
            "YOLOv8n": {
                "model": detection_config["model"],
                "image_size": detection_config["image_size"],
                "batch_size": detection_config["batch_size"],
                "epochs": detection_config["epochs"],
                "patience": detection_config["patience"],
            },
        },
        "artifacts": {
            "classifier_checkpoint": artifact_status(
                CLASSIFIER_CHECKPOINT
            ),
            "detector_checkpoint": artifact_status(
                DETECTOR_CHECKPOINT
            ),
            "classifier_evaluation": artifact_status(
                ROOT
                / "reports"
                / "classification"
                / "evaluation_report.json"
            ),
            "detector_evaluation": artifact_status(
                ROOT
                / "reports"
                / "detection"
                / "evaluation_report.json"
            ),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    MARKDOWN_PATH.write_text(
        build_markdown(summary),
        encoding="utf-8",
    )

    print(f"JSON report: {JSON_PATH}")
    print(f"Markdown report: {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
