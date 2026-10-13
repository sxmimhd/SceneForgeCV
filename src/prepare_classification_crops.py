
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data-source" / "minecraft_mobs_yolo"
OUTPUT = ROOT / "data" / "processed" / "classification"

CLASSES = ["creeper", "skeleton", "spider", "zombie", "enderman"]
MARGIN = 0.12
MIN_SIZE = 8
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    totals = Counter()
    skipped = 0

    for split in ("train", "val"):
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset folders for {split}")

        output_split = OUTPUT / split

        for class_name in CLASSES:
            (output_split / class_name).mkdir(
                parents=True, exist_ok=True
            )

        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                raise FileNotFoundError(f"Missing label: {label_path}")

            lines = [
                line.split()
                for line in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            if not lines:
                continue

            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image_width, image_height = image.size

                for index, parts in enumerate(lines):
                    if len(parts) != 5:
                        skipped += 1
                        continue

                    cls_float, xc, yc, bw, bh = map(float, parts)

                    if (
                        not cls_float.is_integer()
                        or not 0 <= int(cls_float) < len(CLASSES)
                    ):
                        skipped += 1
                        continue

                    cls = int(cls_float)

                    left = (xc - bw / 2) * image_width
                    top = (yc - bh / 2) * image_height
                    right = (xc + bw / 2) * image_width
                    bottom = (yc + bh / 2) * image_height

                    box_width = right - left
                    box_height = bottom - top

                    if box_width < MIN_SIZE or box_height < MIN_SIZE:
                        skipped += 1
                        continue

                    pad_x = box_width * MARGIN
                    pad_y = box_height * MARGIN

                    crop_box = (
                        max(0, round(left - pad_x)),
                        max(0, round(top - pad_y)),
                        min(image_width, round(right + pad_x)),
                        min(image_height, round(bottom + pad_y)),
                    )

                    crop = image.crop(crop_box)

                    if crop.width == 0 or crop.height == 0:
                        skipped += 1
                        continue

                    filename = (
                        f"{image_path.stem}_{index:03d}.jpg"
                    )
                    destination = (
                        output_split / CLASSES[cls] / filename
                    )

                    crop.save(destination, quality=95)
                    totals[(split, CLASSES[cls])] += 1

    print("\nClassification crop summary")
    print("-" * 40)

    for split in ("train", "val"):
        split_total = 0
        print(f"\n{split.upper()}")

        for class_name in CLASSES:
            count = totals[(split, class_name)]
            split_total += count
            print(f"  {class_name:10s}: {count}")

        print(f"  {'TOTAL':10s}: {split_total}")

    print(f"\nSkipped annotations: {skipped}")
    print(f"Output directory: {OUTPUT}")

    if sum(totals.values()) == 0:
        raise RuntimeError("No crops were generated.")

    print("Crop generation completed.")


if __name__ == "__main__":
    main()
