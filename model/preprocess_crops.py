import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

INPUT_DIR = "dataset"
OUTPUT_DIR = "dataset_cropped"
YOLO_MODEL = "yolov8n.pt"
BOX_CONFIDENCE = 0.25


def main():
    model = YOLO(YOLO_MODEL)

    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"Input directory '{INPUT_DIR}' not found")
        sys.exit(1)

    total_images = 0
    total_crops = 0
    skipped = 0

    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing class: {class_name}")

        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue

            total_images += 1
            img = cv2.imread(str(img_file))

            if img is None:
                print(f"  Could not read: {img_file.name}")
                skipped += 1
                continue

            results = model(img, verbose=False)

            crop_idx = 0
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    if label == "bird" and conf > BOX_CONFIDENCE:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img.shape[1], x2)
                        y2 = min(img.shape[0], y2)

                        crop = img[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        out_name = f"{img_file.stem}_crop{crop_idx}.jpg"
                        out_file = output_class_dir / out_name
                        cv2.imwrite(str(out_file), crop)

                        crop_idx += 1
                        total_crops += 1

            if crop_idx == 0:
                skipped += 1

        class_crops = len(list(output_class_dir.iterdir()))
        print(f"  {class_crops} crops saved")

    print()
    print(f"Done! Processed {total_images} images, created {total_crops} crops")
    print(f"Skipped {skipped} images (no bird detected or unreadable)")
    print(f"Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
