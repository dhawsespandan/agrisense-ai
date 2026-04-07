"""
gradcam.py
──────────
Generate Grad-CAM heatmaps to visually explain model predictions.
Shows WHICH region of the apple the model focused on.

Usage
─────
  python gradcam.py --image path/to/apple.jpg
  python gradcam.py --image path/to/apple.jpg --save results/gradcam_out.png
"""

import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEFAULT_MODEL    = "models/apple_disease_efficientnetb0.h5"
IMG_SIZE         = (224, 224)
CLASS_NAMES      = ["Anthracnose", "Black_Pox", "Black_Rot", "Healthy", "Powdery_Mildew"]
LAST_CONV_LAYER  = "top_conv"   # EfficientNet-B0 last conv layer name


def load_model(model_path):
    if not os.path.exists(model_path):
        sys.exit(f"❌  Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def preprocess(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE).astype("float32") / 255.0
    return img_rgb, np.expand_dims(img_resized, 0)


def make_gradcam_heatmap(model, img_tensor, last_conv_layer_name, pred_index=None):
    """Compute Grad-CAM heatmap for the given image tensor."""
    # Build sub-model: inputs → [last_conv_output, predictions]
    try:
        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer(last_conv_layer_name).output, model.output],
        )
    except ValueError:
        # auto-detect last conv layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                grad_model = tf.keras.models.Model(
                    model.inputs,
                    [layer.output, model.output],
                )
                print(f"  ℹ  Using conv layer: {layer.name}")
                break

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colormap        = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_uint8 / 255.0)[..., :3]   # RGB, 0-1
    superimposed    = (original_img / 255.0) * (1 - alpha) + heatmap_colored * alpha
    return np.clip(superimposed, 0, 1)


def run_gradcam(image_path, model_path=DEFAULT_MODEL, save_path=None):
    model   = load_model(model_path)
    orig_rgb, tensor = preprocess(image_path)

    probs     = model.predict(tensor, verbose=0)[0]
    pred_idx  = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]

    heatmap   = make_gradcam_heatmap(model, tensor, LAST_CONV_LAYER, pred_idx)
    overlay   = overlay_gradcam(orig_rgb, heatmap)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#111")

    titles  = ["Original Image", "Grad-CAM Heatmap", "Overlay (Focus Region)"]
    images  = [
        orig_rgb / 255.0,
        plt.cm.jet(heatmap)[..., :3],
        overlay,
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"Predicted: {pred_label.replace('_', ' ')}  |  Confidence: {confidence * 100:.1f}%",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"✔ Grad-CAM saved: {save_path}")
    else:
        plt.show()
    plt.close()

    print(f"\nPredicted : {pred_label}  ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM for Apple Disease Model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--save",  default=None)
    args = parser.parse_args()
    run_gradcam(args.image, args.model, args.save)
