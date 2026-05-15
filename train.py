from pathlib import Path

import torch
import cv2
import albumentations as A
from ultralytics import YOLO

SHOW_AUGMENTATION = True


def preview_augmentation(augmentations: list) -> None:
    out_dir = Path("debug_augmentation")
    out_dir.mkdir(exist_ok=True)

    pipeline = A.Compose(augmentations)
    image_paths = list(Path("self-labled-dataset-yolo/train/images").glob("*.jpg"))

    for img_path in image_paths:
        result = pipeline(image=cv2.imread(str(img_path)))["image"]
        cv2.imwrite(str(out_dir / img_path.name), result)

    print(f"Saved {len(image_paths)} augmented images to {out_dir}/")


def main():
    if torch.cuda.is_available():
        print("NVIDIA GPU (CUDA) detected!")
        compute_device = 'cuda:0'
    elif torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) detected!")
        compute_device = 'mps'
    else:
        print("No compatible GPU found, defaulting to CPU.")
        compute_device = 'cpu'

    model = YOLO('yolo26n.pt')

    augmentations = [
        # flip
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # slight rescaling
        A.RandomScale(scale_range=(-0.1, 0.1), p=0.5),
        # color jitter
        A.RandomBrightnessContrast(brightness_range=(-0.2, 0.2), contrast_range=(-0.2, 0.2), p=0.5),
        A.HueSaturationValue(hue_shift_range=(-10, 10), sat_shift_range=(-20, 20), val_shift_range=(-10, 10), p=0.5),
        # TODO welche augmentations machen sinn?
    ]

    if SHOW_AUGMENTATION:
        preview_augmentation(augmentations)
        return

    print(f"Starting YOLO local training on {compute_device}...")

    model.train(
        data="self-labled-dataset-yolo/data.yaml",
        epochs=100,
        imgsz=1280,
        device=compute_device,
        project="local_bee_models",
        name="yolo26_run_01",
        augmentations=augmentations,
    )


if __name__ == "__main__":
    main()
