import os
import shutil
import random
import yaml
from pathlib import Path

IMAGES_DIR = Path("data/images")
LABELS_DIR = Path("data/labels/labels")
CLASSES_FILE = Path("data/labels/classes.txt")
OUT = Path("dataset")
SPLIT = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42


def get_pairs() -> list[tuple[Path, Path]]:
    """Return (image, label) pairs where both files exist."""
    pairs = []
    for img in sorted(IMAGES_DIR.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = LABELS_DIR / (img.stem + ".txt")
        if label.exists():
            pairs.append((img, label))
        else:
            print(f"  warning: no label for {img.name}, skipping")
    return pairs


def split(pairs: list, ratios: dict[str, float], seed: int) -> dict[str, list]:
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def copy_split(splits: dict[str, list], out: Path) -> None:
    for split_name, pairs in splits.items():
        img_dir = out / split_name / "images"
        lbl_dir = out / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / lbl.name)
        print(f"  {split_name}: {len(pairs)} samples")


def read_classes() -> list[str]:
    if CLASSES_FILE.exists():
        return [l.strip() for l in CLASSES_FILE.read_text().splitlines() if l.strip()]
    import json
    notes = CLASSES_FILE.parent / "notes.json"
    if notes.exists():
        data = json.loads(notes.read_text())
        return [c["name"] for c in data.get("categories", [])]
    return ["object"]


def write_yaml(out: Path, classes: list[str]) -> None:
    config = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
    }
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {yaml_path}")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    pairs = get_pairs()
    print(f"Found {len(pairs)} labeled images across {IMAGES_DIR}")

    splits = split(pairs, SPLIT, SEED)
    copy_split(splits, OUT)

    classes = read_classes()
    write_yaml(OUT, classes)

    print(f"\nDataset ready at '{OUT}' — use '{OUT}/data.yaml' for training.")


if __name__ == "__main__":
    main()
