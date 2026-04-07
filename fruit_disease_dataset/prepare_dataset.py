"""
prepare_dataset.py
──────────────────
Handles this exact folder structure:

Desktop/Apple_Disease_Project/Anthracnose_augmented/
    images/          ← all your .jpg files
    annotations/     ← all your .json files (same filenames as images)

Run this script from inside Apple_Disease_Project folder.
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── CONFIG  ── change these paths if needed ───────────────────────────────────
IMAGES_DIR    = r"Anthracnose_augmented\images"
ANNOT_DIR     = r"Anthracnose_augmented\annotations"
DATASET_DIR   = "dataset"
SPLIT_RATIOS  = (0.70, 0.15, 0.15)   # train / val / test
SEED          = 42

# ── Label mapping (add more diseases here later) ──────────────────────────────
DISEASE_LABELS = {
    "anthracnose"    : "Anthracnose",
    "black_pox"      : "Black_Pox",
    "black pox"      : "Black_Pox",
    "blackpox"       : "Black_Pox",
    "black_rot"      : "Black_Rot",
    "black rot"      : "Black_Rot",
    "blackrot"       : "Black_Rot",
    "powdery_mildew" : "Powdery_Mildew",
    "powdery mildew" : "Powdery_Mildew",
    "powderymildew"  : "Powdery_Mildew",
}
IGNORE_LABELS = {"healthy"}   # whole-apple bounding box — not a disease
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)


def get_class_from_json(json_path: str) -> str:
    with open(json_path, "r") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip().lower()
        if label in IGNORE_LABELS:
            continue
        if label in DISEASE_LABELS:
            return DISEASE_LABELS[label]
        print(f"  ⚠  Unknown label '{label}' in {Path(json_path).name} — treated as Healthy")

    return "Healthy"


def collect_samples(images_dir: str, annot_dir: str):
    images_dir = Path(images_dir)
    annot_dir  = Path(annot_dir)

    class_to_images = defaultdict(list)
    missing_images  = 0

    for json_file in sorted(annot_dir.glob("*.json")):
        stem = json_file.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"  ⚠  No image found for {json_file.name}")
            missing_images += 1
            continue

        cls = get_class_from_json(str(json_file))
        class_to_images[cls].append(str(img_path))

    if missing_images:
        print(f"\n  Total missing images: {missing_images}")

    return class_to_images


def split_list(items, ratios):
    random.shuffle(items)
    n       = len(items)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def copy_files(files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir)


def prepare():
    print("=" * 55)
    print("  Apple Disease — Dataset Preparation")
    print("=" * 55)
    print(f"\n  Images dir      : {IMAGES_DIR}")
    print(f"  Annotations dir : {ANNOT_DIR}")

    if not Path(IMAGES_DIR).exists():
        print(f"\n❌  Images folder not found: {IMAGES_DIR}")
        print("   Make sure you opened VS Code inside Apple_Disease_Project")
        return
    if not Path(ANNOT_DIR).exists():
        print(f"\n❌  Annotations folder not found: {ANNOT_DIR}")
        return

    class_to_images = collect_samples(IMAGES_DIR, ANNOT_DIR)

    total = sum(len(v) for v in class_to_images.values())
    print(f"\n  Total images matched : {total}")
    for cls, imgs in sorted(class_to_images.items()):
        print(f"    {cls:20s}: {len(imgs)} images")

    if total == 0:
        print("\n❌  No matched image+JSON pairs found. Check folder paths above.")
        return

    for cls, imgs in class_to_images.items():
        train_imgs, val_imgs, test_imgs = split_list(imgs, SPLIT_RATIOS)
        copy_files(train_imgs, os.path.join(DATASET_DIR, "train", cls))
        copy_files(val_imgs,   os.path.join(DATASET_DIR, "val",   cls))
        copy_files(test_imgs,  os.path.join(DATASET_DIR, "test",  cls))
        print(f"\n  {cls}")
        print(f"    train={len(train_imgs)}  val={len(val_imgs)}  test={len(test_imgs)}")

    print(f"\n✅  Dataset ready at: {DATASET_DIR}/")
    print("    Now run:  python train.py")


if __name__ == "__main__":
    prepare()
