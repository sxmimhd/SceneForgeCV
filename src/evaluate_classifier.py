
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
CHECKPOINT_PATH = (
    ROOT / "checkpoints" / "classification" / "resnet18_best.pth"
)
REPORT_DIR = ROOT / "reports" / "classification"
REPORT_PATH = REPORT_DIR / "evaluation_report.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {CHECKPOINT_PATH}"
        )

    config = load_config()
    class_names = config["dataset"]["classes"]
    image_size = config["classification"]["image_size"]
    batch_size = config["classification"]["batch_size"]

    val_dir = (
        ROOT
        / config["dataset"]["processed_root"]
        / "classification"
        / "val"
    )

    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Validation crops not found: {val_dir}"
        )

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(
        root=str(val_dir),
        transform=transform,
    )

    if dataset.classes != sorted(class_names):
        raise ValueError(
            "Unexpected validation folder classes.\n"
            f"Found: {dataset.classes}\n"
            f"Expected folder names: {sorted(class_names)}"
        )

    # ImageFolder sorts directory names alphabetically. Map its indices
    # back to the explicit class order used when training the classifier.
    folder_to_config_id = {
        folder_name: class_names.index(folder_name)
        for folder_name in dataset.classes
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["classification"].get("num_workers", 0),
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    incorrect: list[dict] = []
    offset = 0

    with torch.inference_mode():
        for images, folder_labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()

            true_folder_labels = folder_labels.tolist()
            true_labels = [
                folder_to_config_id[dataset.classes[label]]
                for label in true_folder_labels
            ]

            y_true.extend(true_labels)
            y_pred.extend(predictions)

            for batch_index, (true_id, pred_id) in enumerate(
                zip(true_labels, predictions)
            ):
                if true_id != pred_id:
                    sample_path, _ = dataset.samples[
                        offset + batch_index
                    ]
                    incorrect.append(
                        {
                            "image": str(
                                Path(sample_path).relative_to(ROOT)
                            ),
                            "true_class": class_names[true_id],
                            "predicted_class": class_names[pred_id],
                        }
                    )

            offset += len(true_labels)

    labels = list(range(len(class_names)))

    report = {
        "project": "SceneForgeCV",
        "task": "image_classification",
        "model": "ResNet18",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "split": "val",
        "num_samples": len(y_true),
        "class_names": class_names,
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(
                precision_score(
                    y_true, y_pred, labels=labels,
                    average="macro", zero_division=0
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_true, y_pred, labels=labels,
                    average="macro", zero_division=0
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true, y_pred, labels=labels,
                    average="macro", zero_division=0
                )
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist(),
        },
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "incorrect_predictions": incorrect,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Device: {device}")
    print(f"Validation samples: {len(y_true)}")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(f"Macro F1: {report['metrics']['macro_f1']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    print(f"Incorrect predictions: {len(incorrect)}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
