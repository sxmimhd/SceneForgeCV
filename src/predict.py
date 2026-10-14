
import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Minecraft mob detection with SceneForgeCV."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image or directory of images.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence (0 to 1).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    weights = Path(args.weights).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Image source not found: {source}")

    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1.")

    model = YOLO(str(weights))

    output_dir = ROOT / "runs" / "predictions"

    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="minecraft_mobs",
        exist_ok=True,
        verbose=False,
    )

    print(f"\nModel: {weights}")
    print(f"Source: {source}")

    for result in results:
        print(f"\nImage: {result.path}")
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print("  No mobs detected.")
            continue

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = model.names[class_id]

            print(
                f"  {class_name}: {confidence:.3f} "
                f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
            )

    print(f"\nAnnotated results: {output_dir / 'minecraft_mobs'}")


if __name__ == "__main__":
    main()
