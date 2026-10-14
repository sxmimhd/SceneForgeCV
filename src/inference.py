
from pathlib import Path
from typing import Any

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WEIGHTS = (
    ROOT
    / "runs"
    / "detection"
    / "yolov8n_minecraft_mobs"
    / "weights"
    / "best.pt"
)


class SceneForgeDetector:
    """Reusable inference interface for the SceneForgeCV detector."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        confidence: float = 0.25,
        image_size: int = 640,
        device: str | int | None = None,
    ) -> None:
        self.weights = Path(weights).resolve()

        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Detector weights not found: {self.weights}"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")

        self.confidence = confidence
        self.image_size = image_size
        self.device = device

        self.model = YOLO(str(self.weights))
        self.class_names = self.model.names

    def predict(
        self,
        source: str | Path,
    ) -> list[dict[str, Any]]:
        """Detect mobs in an image and return structured predictions."""
        source_path = Path(source).resolve()

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Input image not found: {source_path}"
            )

        results = self.model.predict(
            source=str(source_path),
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        predictions: list[dict[str, Any]] = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                predictions.append(
                    {
                        "class_id": class_id,
                        "class_name": self.class_names[class_id],
                        "confidence": confidence,
                        "bbox_xyxy": [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ],
                    }
                )

        return predictions
