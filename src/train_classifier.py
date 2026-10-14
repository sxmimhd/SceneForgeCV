
import json
import random
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OUTPUT = ROOT / "reports" / "classification"
CHECKPOINT_DIR = ROOT / "checkpoints" / "classification"


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = config["classification"]
    classes = config["dataset"]["classes"]
    seed = config["project"]["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    learning_rate = float(cfg["learning_rate"])
    num_workers = int(cfg.get("num_workers", 0))

    train_transform = Compose([
        Resize((image_size, image_size)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    data_root = ROOT / config["dataset"]["processed_root"] / "classification"
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            "Classification crops not found. Run "
            "src/prepare_classification_crops.py first."
        )

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    if train_ds.classes != val_ds.classes:
        raise ValueError(
            f"Train/validation class mismatch: "
            f"{train_ds.classes} vs {val_ds.classes}"
        )

    if set(train_ds.classes) != set(classes):
        raise ValueError(
            f"Configured classes {classes} do not match "
            f"dataset classes {train_ds.classes}"
        )

    # Use the configured class order throughout the project.
    class_to_index = {
        name: index for index, name in enumerate(classes)
    }

    train_ds.classes = classes
    train_ds.class_to_idx = class_to_index
    train_ds.samples = [
        (path, class_to_index[Path(path).parent.name])
        for path, _ in train_ds.samples
    ]
    train_ds.targets = [label for _, label in train_ds.samples]

    val_ds.classes = classes
    val_ds.class_to_idx = class_to_index
    val_ds.samples = [
        (path, class_to_index[Path(path).parent.name])
        for path, _ in val_ds.samples
    ]
    val_ds.targets = [label for _, label in val_ds.samples]

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Training or validation dataset is empty.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Train crops: {len(train_ds)}")
    print(f"Validation crops: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")

    class_counts = np.bincount(
        [label for _, label in train_ds.samples],
        minlength=len(classes),
    )
    class_weights = (
        class_counts.sum() / np.maximum(class_counts, 1)
    ) ** 0.5
    class_weights = class_weights / class_weights.mean()

    weights = ResNet18_Weights.DEFAULT if cfg.get("pretrained", True) else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / "resnet18_best.pth"

    best_accuracy = -1.0
    history = []
    best_predictions = None
    best_targets = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_targets, train_predictions = [], []

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_targets.extend(targets.detach().cpu().tolist())
            train_predictions.extend(
                logits.argmax(dim=1).detach().cpu().tolist()
            )

        train_loss = running_loss / len(train_ds)
        train_accuracy = accuracy_score(
            train_targets, train_predictions
        )

        model.eval()
        val_loss_sum = 0.0
        val_targets, val_predictions = [], []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, targets)

                val_loss_sum += loss.item() * images.size(0)
                val_targets.extend(targets.cpu().tolist())
                val_predictions.extend(logits.argmax(1).cpu().tolist())

        val_loss = val_loss_sum / len(val_ds)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        scheduler.step(val_accuracy)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": float(train_accuracy),
            "val_loss": val_loss,
            "val_accuracy": float(val_accuracy),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_predictions = val_predictions.copy()
            best_targets = val_targets.copy()

            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "image_size": image_size,
                "epoch": epoch,
                "val_accuracy": float(val_accuracy),
                "architecture": "resnet18",
            }, best_path)

    (OUTPUT / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    report = classification_report(
        best_targets,
        best_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    (OUTPUT / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    cm = confusion_matrix(
        best_targets, best_predictions, labels=list(range(len(classes)))
    )
    fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Count: %{z}<extra></extra>"
        ),
    )
    )

    fig.update_layout(
        title="ResNet-18 — Validation Confusion Matrix",
        xaxis_title="Predicted class",
        yaxis_title="True class",
        template="plotly_dark",
    )

    fig.write_html(
        str(OUTPUT / "confusion_matrix.html"),
        include_plotlyjs=True,
    )
    fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Training and Validation Loss", "Accuracy"),
    )

    epochs_list = [item["epoch"] for item in history]

    fig.add_trace(
        go.Scatter(
            x=epochs_list,
            y=[item["train_loss"] for item in history],
            mode="lines+markers",
            name="Train Loss",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs_list,
            y=[item["val_loss"] for item in history],
            mode="lines+markers",
            name="Validation Loss",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs_list,
            y=[item["train_accuracy"] for item in history],
            mode="lines+markers",
            name="Train Accuracy",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs_list,
            y=[item["val_accuracy"] for item in history],
            mode="lines+markers",
            name="Validation Accuracy",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="SceneForge CV — ResNet-18 Training",
        template="plotly_dark",
        height=500,
    )

    fig.write_html(
        str(OUTPUT / "training_curves.html"),
        include_plotlyjs=True,
    )
    summary = {
        "best_val_accuracy": float(best_accuracy),
        "best_epoch": int(
            max(history, key=lambda item: item["val_accuracy"])["epoch"]
        ),
        "train_crops": len(train_ds),
        "val_crops": len(val_ds),
        "class_counts_train": {
            name: int(class_counts[i]) for i, name in enumerate(classes)
        },
        "classification_report": report,
        "checkpoint": str(best_path.relative_to(ROOT)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nBest validation accuracy:", f"{best_accuracy:.4f}")
    print("Checkpoint:", best_path)
    print("Reports:", OUTPUT)


if __name__ == "__main__":
    main()
