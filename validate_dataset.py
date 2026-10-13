
from collections import Counter
from hashlib import sha256
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data-source" / "minecraft_mobs_yolo"
SPLITS = ("train", "val")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NUM_CLASSES = 5
TOLERANCE = 1e-6


def main():
    errors = []
    total_images = 0
    total_boxes = 0
    empty_images = 0
    class_counts = Counter()
    hashes = {}

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: missing images or labels directory")
            continue

        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_stems = {p.stem for p in images}
        label_files = list(label_dir.glob("*.txt"))
        label_stems = {p.stem for p in label_files}

        for stem in sorted(image_stems - label_stems):
            errors.append(f"{split}: missing label for {stem}")

        for stem in sorted(label_stems - image_stems):
            errors.append(f"{split}: label without image: {stem}")

        split_boxes = 0
        split_empty = 0

        for image_path in images:
            total_images += 1

            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    image_width, image_height = image.size
                    image.load()
            except Exception as exc:
                errors.append(f"Unreadable image {image_path.name}: {exc}")
                continue

            digest = sha256(image_path.read_bytes()).hexdigest()
            if digest in hashes:
                print(
                    f"Exact duplicate: {image_path} "
                    f"== {hashes[digest]}"
                )
            else:
                hashes[digest] = image_path

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            lines = [
                line.strip()
                for line in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            if not lines:
                empty_images += 1
                split_empty += 1
                continue

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()

                try:
                    if len(parts) != 5:
                        raise ValueError("expected 5 values")

                    cls_float, xc, yc, width, height = map(float, parts)

                    if not cls_float.is_integer():
                        raise ValueError("class ID must be an integer")

                    cls = int(cls_float)

                    if not 0 <= cls < NUM_CLASSES:
                        raise ValueError(f"invalid class ID {cls}")

                    if not all(
                        0 <= value <= 1
                        for value in (xc, yc, width, height)
                    ):
                        raise ValueError("normalized values outside [0, 1]")

                    if width <= 0 or height <= 0:
                        raise ValueError("box width and height must be positive")

                    left = xc - width / 2
                    top = yc - height / 2
                    right = xc + width / 2
                    bottom = yc + height / 2

                    if (
                        left < -TOLERANCE
                        or top < -TOLERANCE
                        or right > 1 + TOLERANCE
                        or bottom > 1 + TOLERANCE
                    ):
                        raise ValueError("box extends outside image boundaries")

                    # Access dimensions so image decoding is part of validation.
                    if image_width <= 0 or image_height <= 0:
                        raise ValueError("invalid image dimensions")

                    class_counts[cls] += 1
                    total_boxes += 1
                    split_boxes += 1

                except ValueError as exc:
                    errors.append(
                        f"{label_path.name}:{line_number}: {exc}"
                    )

        print(
            f"{split.upper()}: images={len(images)}, "
            f"boxes={split_boxes}, empty_labels={split_empty}"
        )

    class_names = ["creeper", "skeleton", "spider", "zombie", "enderman"]

    print("\n--- DATASET VALIDATION ---")
    print("Total images:", total_images)
    print("Total valid boxes:", total_boxes)
    print("Empty label files:", empty_images)
    print("Class counts:")
    for class_id, name in enumerate(class_names):
        print(f"  {name}: {class_counts[class_id]}")

    print("Errors:", len(errors))
    for error in errors[:50]:
        print("ERROR:", error)

    if len(errors) > 50:
        print(f"... and {len(errors) - 50} more errors")

    if errors:
        raise SystemExit("Validation failed. Fix issues before training.")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
