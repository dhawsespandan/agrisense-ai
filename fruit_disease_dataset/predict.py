"""
predict.py
──────────
Run inference on a single image or a folder of images.

Usage
─────
  # single image
  python predict.py --image path/to/apple.jpg

  # entire folder
  python predict.py --folder path/to/test_images/

  # custom model / threshold
  python predict.py --image apple.jpg --model models/apple_disease_efficientnetb0.h5 --threshold 0.6
"""

import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL     = "models/apple_disease_efficientnetb0.h5"
IMG_SIZE          = (224, 224)
CLASS_NAMES       = ["Anthracnose", "Black_Pox", "Black_Rot", "Healthy", "Powdery_Mildew"]
CONFIDENCE_THRESH = 0.50   # below this → "Low confidence / Uncertain"

# Colour palette for each class (BGR for OpenCV, RGB for matplotlib)
CLASS_COLORS = {
    "Anthracnose"    : (0.82, 0.19, 0.19),   # red
    "Black_Pox"      : (0.25, 0.25, 0.25),   # dark grey
    "Black_Rot"      : (0.55, 0.10, 0.55),   # purple
    "Healthy"        : (0.13, 0.63, 0.23),   # green
    "Powdery_Mildew" : (0.15, 0.50, 0.80),   # blue
}
# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        sys.exit(f"❌  Model not found: {model_path}\n"
                 f"   Train first with:  python train.py")
    print(f"✔  Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)


def preprocess(image_path: str) -> np.ndarray:
    """Load, resize and normalise an image for the model."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized.astype("float32") / 255.0
    return img_rgb, np.expand_dims(img_norm, 0)


def predict_single(model, image_path: str, threshold: float = CONFIDENCE_THRESH):
    """Return (class_name, confidence, all_probs)."""
    original, tensor = preprocess(image_path)
    probs = model.predict(tensor, verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    label = CLASS_NAMES[idx] if conf >= threshold else "Uncertain"
    return label, conf, probs, original


def visualise(image_path: str, label: str, conf: float, probs: np.ndarray,
              original_img: np.ndarray, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # ── Left: original image with overlay ────────────────────────────────────
    ax_img = axes[0]
    ax_img.imshow(original_img)
    ax_img.axis("off")

    colour = CLASS_COLORS.get(label, (1, 0.6, 0))
    rect = mpatches.FancyBboxPatch(
        (5, 5), original_img.shape[1] - 10, original_img.shape[0] - 10,
        boxstyle="round,pad=3", linewidth=4,
        edgecolor=colour, facecolor="none",
    )
    ax_img.add_patch(rect)

    disease_text  = label.replace("_", " ")
    confidence_pct = f"{conf * 100:.1f}%"
    ax_img.set_title(
        f"Predicted: {disease_text}\nConfidence: {confidence_pct}",
        fontsize=14, fontweight="bold", color="white", pad=10,
    )

    # ── Right: probability bar chart ─────────────────────────────────────────
    ax_bar = axes[1]
    ax_bar.set_facecolor("#16213e")

    y_pos  = np.arange(len(CLASS_NAMES))
    colors = [CLASS_COLORS.get(c, (0.5, 0.5, 0.5)) for c in CLASS_NAMES]
    bars   = ax_bar.barh(y_pos, probs * 100, color=colors, edgecolor="white",
                         linewidth=0.5, height=0.6)

    for bar, prob in zip(bars, probs):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{prob * 100:.1f}%", va="center", ha="left",
                    color="white", fontsize=10)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([c.replace("_", " ") for c in CLASS_NAMES],
                           color="white", fontsize=11)
    ax_bar.set_xlabel("Confidence (%)", color="white", fontsize=11)
    ax_bar.set_title("Class Probabilities", color="white", fontsize=13,
                     fontweight="bold")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#444")
    ax_bar.set_xlim(0, 115)
    ax_bar.grid(axis="x", alpha=0.2, color="white")

    plt.tight_layout(pad=2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ✔ Result image saved: {save_path}")
    else:
        plt.show()
    plt.close()


def predict_folder(model, folder: str, threshold: float, save_dir: str = "results/predictions"):
    os.makedirs(save_dir, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in Path(folder).iterdir() if p.suffix in exts]

    if not images:
        print(f"❌  No images found in: {folder}")
        return

    print(f"\nPredicting {len(images)} images …\n")
    results = []

    for img_path in sorted(images):
        label, conf, probs, orig = predict_single(model, str(img_path), threshold)
        save_path = os.path.join(save_dir, f"pred_{img_path.stem}.png")
        visualise(str(img_path), label, conf, probs, orig, save_path)
        results.append((img_path.name, label, conf))
        print(f"  {img_path.name:40s}  →  {label:18s}  ({conf * 100:.1f}%)")

    print(f"\n✅ All predictions saved to: {save_dir}/")
    return results


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Apple Disease Detection — Inference"
    )
    parser.add_argument("--image",     type=str, help="Path to a single image")
    parser.add_argument("--folder",    type=str, help="Path to a folder of images")
    parser.add_argument("--model",     type=str, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESH,
                        help="Minimum confidence to assign a label (default 0.5)")
    parser.add_argument("--save",      type=str, default=None,
                        help="(single image) path to save result PNG")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help()
        sys.exit("\n❌  Provide --image or --folder")

    model = load_model(args.model)

    if args.image:
        label, conf, probs, orig = predict_single(model, args.image, args.threshold)
        print("\n" + "=" * 45)
        print(f"  Image     : {args.image}")
        print(f"  Predicted : {label.replace('_', ' ')}")
        print(f"  Confidence: {conf * 100:.2f}%")
        print("=" * 45)
        print("\nAll class probabilities:")
        for cls, p in zip(CLASS_NAMES, probs):
            bar = "█" * int(p * 30)
            print(f"  {cls:20s}: {p * 100:5.1f}%  {bar}")
        visualise(args.image, label, conf, probs, orig, save_path=args.save)

    elif args.folder:
        predict_folder(model, args.folder, args.threshold)


if __name__ == "__main__":
    main()
