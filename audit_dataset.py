from pathlib import Path
from collections import Counter

root = Path("data-source/minecraft_mobs_yolo")
classes = ["creeper", "skeleton", "spider", "zombie", "enderman"]
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for split in ("train", "val"):
    image_dir = root / split / "images"
    label_dir = root / split / "labels"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    labels = {p.stem: p for p in label_dir.glob("*.txt")}
    image_stems = {p.stem for p in images}

    missing = sorted(image_stems - labels.keys())
    orphaned = sorted(labels.keys() - image_stems)
    class_counts = Counter()
    invalid = []
    empty_labels = 0
    total_boxes = 0

    for label_path in labels.values():
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
        rows = [line.strip() for line in lines if line.strip()]

        if not rows:
            empty_labels += 1

        for line_number, line in enumerate(rows, start=1):
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError("expected 5 values")

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if not 0 <= class_id < len(classes):
                    raise ValueError("class ID outside 0-4")

                if not all(0 <= x <= 1 for x in coords):
                    raise ValueError("coordinate outside [0, 1]")

                _, _, width, height = coords
                if width <= 0 or height <= 0:
                    raise ValueError("non-positive box size")

                class_counts[classes[class_id]] += 1
                total_boxes += 1

            except (ValueError, OverflowError) as exc:
                invalid.append(
                    f"{label_path.name}:{line_number}: {exc} | {line}"
                )

    print(f"\n===== {split.upper()} =====")
    print(f"Images: {len(images)}")
    print(f"Label files: {len(labels)}")
    print(f"Images missing labels: {len(missing)}")
    print(f"Labels without images: {len(orphaned)}")
    print(f"Empty label files: {empty_labels}")
    print(f"Valid bounding boxes: {total_boxes}")
    print(f"Invalid annotation rows: {len(invalid)}")
    print("Class counts:", dict(class_counts))

    if missing:
        print("Missing label examples:", missing[:5])
    if orphaned:
        print("Orphaned label examples:", orphaned[:5])
    if invalid:
        print("Invalid row examples:")
        for issue in invalid[:10]:
            print(" ", issue)

print("\nAudit complete. No files were modified.")
