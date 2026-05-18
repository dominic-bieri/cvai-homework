import logging
from pathlib import Path

import albumentations as A
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAIN_IMAGES = Path("self-labled-dataset-yolo/train/images")
TRAIN_LABELS = Path("self-labled-dataset-yolo/train/labels")

# Total dataset size relative to the original training set.
# 2.0 = 2x total  |  1.25 = 25% more  |  4.0 = 4x total
DATASET_MULTIPLIER: float = 1.25

pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(angle_range=(-30, 30), p=0.5),
        A.RandomBrightnessContrast(brightness_range=(-0.3, 0.3), contrast_range=(-0.3, 0.3), p=0.6),
        A.HueSaturationValue(hue_shift_range=(-10, 10), sat_shift_range=(-30, 30), val_shift_range=(-20, 20), p=0.4),
        A.RandomShadow(p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
        A.MotionBlur(blur_range=(3, 7), p=0.3),
        A.GaussianBlur(blur_range=(3, 5), p=0.2),
        A.GaussNoise(p=0.3),
        A.ImageCompression(quality_range=(60, 100), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(20, 60), hole_width_range=(20, 60), p=0.3),
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ),
)


def read_labels(path: Path) -> tuple[list[int], list[list[float]]]:
    class_ids, bboxes = [], []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        class_ids.append(int(parts[0]))
        bboxes.append([float(x) for x in parts[1:5]])
    return class_ids, bboxes


def write_labels(path: Path, class_ids: list, bboxes: list) -> None:
    lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in zip(class_ids, bboxes)]
    path.write_text("\n".join(lines))  # empty string for background images — correct YOLO format


def main() -> None:
    image_paths = [p for p in sorted(TRAIN_IMAGES.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    originals = [p for p in image_paths if "_aug" not in p.stem]

    if len(originals) < len(image_paths):
        logger.error("Augmented files already exist — delete self-labled-dataset-yolo/ and re-run create_dataset.py first.")
        return

    n_aug = round((DATASET_MULTIPLIER - 1) * len(originals))
    logger.info("%d training images → generating %d augmented copies (%.2fx total)", len(originals), n_aug, DATASET_MULTIPLIER)

    aug_count = 0
    for i in range(n_aug):
        img_path = originals[i % len(originals)]
        lbl_path = TRAIN_LABELS / (img_path.stem + ".txt")

        if not lbl_path.exists():
            logger.warning("No label for %s, skipping", img_path.name)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read %s, skipping", img_path.name)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        class_ids, bboxes = read_labels(lbl_path)
        is_background = len(bboxes) == 0

        result = pipeline(image=img, bboxes=bboxes, class_labels=class_ids)

        if not is_background and len(result["bboxes"]) == 0:
            continue

        stem = f"{img_path.stem}_aug{i}"
        out_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(TRAIN_IMAGES / f"{stem}.jpg"), out_img)
        write_labels(TRAIN_LABELS / f"{stem}.txt", result["class_labels"], result["bboxes"])
        aug_count += 1

        if aug_count % 50 == 0:
            logger.info("Progress: %d/%d", aug_count, n_aug)

    logger.info("Done — %d originals + %d augmented = %d training images", len(originals), aug_count, len(originals) + aug_count)


if __name__ == "__main__":
    main()
