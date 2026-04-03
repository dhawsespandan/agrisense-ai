"""
models/leaf_disease/prepare_data.py
=====================================
Data Engineering Pipeline — Apple Leaf Disease Detection

Reads from the original zip archive (source data is NEVER modified).
Produces:
    processed_dataset/
        train/   ← TARGET_TRAIN images per class (originals + offline augmentation)
        val/     ← 15% of each class (originals only, no augmentation)
        test/    ← 15% of each class (originals only, no augmentation)

Usage
-----
    # From the model-service/ directory:
    python -m models.leaf_disease.prepare_data \
        --zip_path /path/to/archive.zip \
        --out_dir  /path/to/processed_dataset

    # With defaults:
    python -m models.leaf_disease.prepare_data

Source class distribution (approximate)
-----------------------------------------
    Alternaria leaf spot : 278   ← minority
    Brown spot           : 215   ← heaviest augmentation needed
    Gray spot            : 395
    Healthy leaf         : 409
    Rust                 : 344

Target after pipeline
----------------------
    TARGET_TRAIN images per class in train/
    ~15% of originals in val/ and test/ (unmodified — no augmentation)

Augmentation pipeline (Albumentations)
---------------------------------------
    HorizontalFlip, Rotate(±20°)
    RandomBrightnessContrast  — exposure/contrast variance (mobile cameras)
    ColorJitter               — hue/saturation shift; prevents colour-only reliance
    RandomShadow              — simulates shadows cast by leaves/branches in field
    GaussNoise, Blur          — sensor noise and motion blur
    RandomResizedCrop         — zoom-in simulation for small lesions
"""

import argparse
import io
import os
import random
import shutil
import zipfile
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────

SEED          = 42
TARGET_TRAIN  = 600
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15

CLASS_MAP = {
    "Alternaria leaf spot" : "alternaria_leaf_spot",
    "Brown spot"           : "brown_spot",
    "Gray spot"            : "gray_spot",
    "Healthy leaf"         : "healthy_leaf",
    "Rust"                 : "rust",
}

DEFAULT_ZIP = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "attached_assets", "archive_(3)_1775155146196.zip",
))
DEFAULT_OUT = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "processed_dataset",
))


# ── Augmentation pipeline ──────────────────────────────────────────────────
# Each transform targets a specific real-world degradation:
#   ColorJitter    — colour temperature and saturation shifts (sunlight angle,
#                    white balance errors in mobile cameras, different devices)
#   RandomShadow   — partial shadow from leaves, branches or photographer's body;
#                    the most common cause of misclassification on field images
#   RandomBrightnessContrast — overall exposure variance (overcast vs sunny)
#   GaussNoise/Blur          — sensor noise and motion blur

_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.7),
    A.RandomBrightnessContrast(
        brightness_limit=0.3, contrast_limit=0.3, p=0.7
    ),
    A.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.3, hue=0.08, p=0.6
    ),
    A.RandomShadow(
        shadow_roi=(0, 0.0, 1, 1),
        num_shadows_limit=(1, 2),
        shadow_dimension=5,
        p=0.4,
    ),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.4),
    A.Blur(blur_limit=3, p=0.3),
    A.RandomResizedCrop(
        size=(224, 224), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.5
    ),
    A.Resize(224, 224),
])


# ── Helpers ────────────────────────────────────────────────────────────────

def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _augment_image(pil_img: Image.Image) -> Image.Image:
    bgr    = _pil_to_cv(pil_img)
    result = _AUG(image=bgr)["image"]
    return _cv_to_pil(result)


def _save_pil(pil_img: Image.Image, path: str) -> None:
    pil_img.save(path, quality=95)


def _load_from_zip(zf: zipfile.ZipFile, name: str) -> Image.Image:
    with zf.open(name) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def _split_indices(
    n: int,
    train_r: float = TRAIN_RATIO,
    val_r: float   = VAL_RATIO,
    seed: int      = SEED,
) -> tuple[list[int], list[int], list[int]]:
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ── Main pipeline ──────────────────────────────────────────────────────────

def run(zip_path: str, out_dir: str) -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    split_root = Path(out_dir)
    for split in ("train", "val", "test"):
        for cls_dir in CLASS_MAP.values():
            (split_root / split / cls_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nReading dataset from : {zip_path}")
    print(f"Output directory     : {out_dir}\n")

    with zipfile.ZipFile(zip_path) as zf:
        class_files: dict[str, list[str]] = {k: [] for k in CLASS_MAP}
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            top = name.split("/")[0]
            if top in CLASS_MAP:
                class_files[top].append(name)

        for zip_cls, files in sorted(class_files.items()):
            norm_cls = CLASS_MAP[zip_cls]
            n        = len(files)
            print(f"  [{zip_cls}]  source images: {n}")

            train_idx, val_idx, test_idx = _split_indices(n)

            for split_name, split_idx in (("val", val_idx), ("test", test_idx)):
                dest_dir = split_root / split_name / norm_cls
                for i, idx in enumerate(split_idx):
                    img = _load_from_zip(zf, files[idx])
                    _save_pil(img, str(dest_dir / f"orig_{i:04d}.jpg"))
                print(f"    {split_name:5s}: {len(split_idx):3d} images (originals, no augmentation)")

            train_dir  = split_root / "train" / norm_cls
            train_imgs = []
            for i, idx in enumerate(train_idx):
                img  = _load_from_zip(zf, files[idx])
                path = str(train_dir / f"orig_{i:04d}.jpg")
                _save_pil(img, path)
                train_imgs.append(img)

            n_orig   = len(train_imgs)
            n_needed = TARGET_TRAIN - n_orig
            print(f"    train: {n_orig:3d} originals  →  need {n_needed} augmented")

            aug_count  = 0
            source_cyc = list(range(n_orig))
            random.shuffle(source_cyc)
            cycle_pos  = 0

            while aug_count < n_needed:
                src_img  = train_imgs[source_cyc[cycle_pos % len(source_cyc)]]
                cycle_pos += 1
                aug_img  = _augment_image(src_img)
                path     = str(train_dir / f"aug_{aug_count:05d}.jpg")
                _save_pil(aug_img, path)
                aug_count += 1

            total_train = n_orig + aug_count
            print(f"    train: {total_train} total  ({n_orig} orig + {aug_count} augmented)")

    print(f"\nDone. Split saved to: {split_root}")
    print("Class breakdown (train):")
    for cls_dir in sorted(CLASS_MAP.values()):
        count = len(list((split_root / "train" / cls_dir).glob("*.jpg")))
        print(f"  {cls_dir}: {count}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build balanced train/val/test splits with offline augmentation"
    )
    parser.add_argument(
        "--zip_path",
        default=DEFAULT_ZIP,
        help="Path to the original dataset zip archive (never modified)",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT,
        help="Root output directory for the processed dataset "
             "(will contain train/, val/, test/)",
    )
    args = parser.parse_args()
    run(args.zip_path, args.out_dir)
