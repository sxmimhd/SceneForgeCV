
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data-source" / "minecraft_mobs_yolo"
OUTPUT = ROOT / "reports" / "figures" / "train_contact_sheet.jpg"

CLASS_NAMES = ["creeper", "skeleton", "spider", "zombie", "enderman"]
IMAGE_SIZE = (320, 220)
COLS = 4
ANNOTATED_SAMPLES = 18
BACKGROUND_SAMPLES = 6
SEED = 42


def read_labels(label_path):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        cls, xc, yc, width, height = map(float, parts)
        boxes.append((int(cls), xc, yc, width, height))

    return boxes


def main():
    random.seed(SEED)

    image_dir = DATASET / "train" / "images"
    label_dir = DATASET / "train" / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders under {DATASET}"
        )

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )

    annotated, background = [], []

    for path in images:
        label_path = label_dir / f"{path.stem}.txt"
        boxes = read_labels(label_path)

        if boxes:
            annotated.append((path, boxes))
        else:
            background.append((path, boxes))

    random.shuffle(annotated)
    random.shuffle(background)

    samples = (
        annotated[:ANNOTATED_SAMPLES]
        + background[:BACKGROUND_SAMPLES]
    )
    random.shuffle(samples)

    cell_w, cell_h = IMAGE_SIZE
    rows = (len(samples) + COLS - 1) // COLS
    sheet = Image.new(
        "RGB", (COLS * cell_w, rows * cell_h), "#20202a"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (path, boxes) in enumerate(samples):
        col, row = index % COLS, index // COLS
        px, py = col * cell_w, row * cell_h

        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                original_w, original_h = image.size
                image.thumbnail((cell_w - 12, cell_h - 42))
        except Exception as exc:
            draw.text(
                (px + 6, py + 6),
                f"IMAGE ERROR: {path.name}",
                fill="red",
                font=font,
            )
            print(f"Could not read {path}: {exc}")
            continue

        image_x = px + (cell_w - image.width) // 2
        image_y = py + 24

        sheet.paste(image, (image_x, image_y))

        scale_x = image.width / original_w
        scale_y = image.height / original_h

        for cls, xc, yc, bw, bh in boxes:
            if not 0 <= cls < len(CLASS_NAMES):
                continue

            left = image_x + (xc - bw / 2) * original_w * scale_x
            top = image_y + (yc - bh / 2) * original_h * scale_y
            right = image_x + (xc + bw / 2) * original_w * scale_x
            bottom = image_y + (yc + bh / 2) * original_h * scale_y

            draw.rectangle(
                (left, top, right, bottom),
                outline="#55ff88",
                width=2,
            )
            draw.text(
                (left, max(image_y, top - 12)),
                CLASS_NAMES[cls],
                fill="#55ff88",
                font=font,
            )

        label = f"{path.name[:27]} | {len(boxes)} objects"
        if not boxes:
            label = f"{path.name[:27]} | BACKGROUND"

        draw.text(
            (px + 5, py + 5),
            label,
            fill="white",
            font=font,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT, quality=92)

    print(f"Images scanned: {len(images)}")
    print(f"Annotated images: {len(annotated)}")
    print(f"Background images: {len(background)}")
    print(f"Samples displayed: {len(samples)}")
    print(f"Contact sheet saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
