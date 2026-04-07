"""
Apple Disease Detection - Training Script
Model: EfficientNet-B0 with Transfer Learning
Auto-detects number of classes from your dataset folder.
Works with 1 class now, and 5 classes later automatically.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "img_size": (224, 224),
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-4,
    "fine_tune_lr": 1e-5,
    "fine_tune_at": 100,
    "fine_tune_epochs": 20,
    "data_dir": "dataset",
    "model_save_path": "models/apple_disease_efficientnetb0.h5",
    "history_save_path": "models/training_history.json",
    "seed": 42,
}

tf.random.set_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─────────────────────────────────────────────
# AUTO-DETECT CLASSES FROM DATASET FOLDER
# ─────────────────────────────────────────────
def detect_classes(data_dir):
    train_dir = os.path.join(data_dir, "train")
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return classes


# ─────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────
def build_data_generators(data_dir, img_size, batch_size, num_classes):
    # binary mode for 1 class, categorical for 2+
    class_mode = "binary" if num_classes == 1 else "categorical"

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20.0,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size, batch_size=batch_size,
        class_mode=class_mode, shuffle=True, seed=CONFIG["seed"],
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=img_size, batch_size=batch_size,
        class_mode=class_mode, shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size, batch_size=batch_size,
        class_mode=class_mode, shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes, img_size, learning_rate):
    inputs = layers.Input(shape=(*img_size, 3))
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "categorical_crossentropy"

    model = Model(inputs, outputs, name="AppleDisease_EfficientNetB0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_for_fine_tuning(model, base, fine_tune_at, lr, num_classes):
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    loss = "binary_crossentropy" if num_classes == 1 else "categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss, metrics=["accuracy"],
    )
    print(f"\n✔ Fine-tuning enabled: {sum(l.trainable for l in model.layers)} trainable layers")


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        ModelCheckpoint(model_path, monitor="val_accuracy",
                        save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
    ]


# ─────────────────────────────────────────────
# PLOTS & EVALUATION
# ─────────────────────────────────────────────
def plot_training_history(history, fine_history=None, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    acc   = history.history["accuracy"] + (fine_history.history["accuracy"] if fine_history else [])
    val_a = history.history["val_accuracy"] + (fine_history.history["val_accuracy"] if fine_history else [])
    loss  = history.history["loss"] + (fine_history.history["loss"] if fine_history else [])
    val_l = history.history["val_loss"] + (fine_history.history["val_loss"] if fine_history else [])

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc,   label="Train Accuracy", color="#27ae60")
    plt.plot(val_a, label="Val Accuracy",   color="#e74c3c", linestyle="--")
    if fine_history:
        plt.axvline(len(history.history["accuracy"]) - 1, color="gray", linestyle=":", label="Fine-tune start")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss,  label="Train Loss", color="#27ae60")
    plt.plot(val_l, label="Val Loss",   color="#e74c3c", linestyle="--")
    if fine_history:
        plt.axvline(len(history.history["loss"]) - 1, color="gray", linestyle=":", label="Fine-tune start")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"✔ Training curves saved → results/training_curves.png")


def evaluate_model(model, test_gen, class_names, num_classes, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 50)
    print("EVALUATION ON TEST SET")
    print("=" * 50)

    results = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss     : {results[0]:.4f}")
    print(f"Test Accuracy : {results[1] * 100:.2f}%")

    preds_raw = model.predict(test_gen, verbose=1)
    if num_classes == 1:
        y_pred = (preds_raw > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(preds_raw, axis=1)

    y_true = test_gen.classes
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(class_names) * 2), max(5, len(class_names) * 1.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"✔ Confusion matrix saved → results/confusion_matrix.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Apple Disease Detection — EfficientNet-B0")
    print("=" * 60)

    # Auto-detect
    class_names = detect_classes(CONFIG["data_dir"])
    num_classes = len(class_names)
    print(f"\n✔ Classes detected ({num_classes}): {class_names}")

    train_gen, val_gen, test_gen = build_data_generators(
        CONFIG["data_dir"], CONFIG["img_size"], CONFIG["batch_size"], num_classes
    )
    print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

    model, base = build_model(num_classes, CONFIG["img_size"], CONFIG["learning_rate"])
    model.summary()

    callbacks = get_callbacks(CONFIG["model_save_path"])

    # Phase 1: warm-up
    print("\n▶ Phase 1: Warm-up (base frozen) …")
    history = model.fit(train_gen, epochs=CONFIG["epochs"],
                        validation_data=val_gen, callbacks=callbacks, verbose=1)

    # Phase 2: fine-tune
    print("\n▶ Phase 2: Fine-tuning …")
    unfreeze_for_fine_tuning(model, base, CONFIG["fine_tune_at"],
                             CONFIG["fine_tune_lr"], num_classes)
    fine_history = model.fit(train_gen, epochs=CONFIG["fine_tune_epochs"],
                             validation_data=val_gen, callbacks=callbacks, verbose=1)

    # Save history
    combined = {k: history.history[k] + fine_history.history[k] for k in history.history}
    os.makedirs(os.path.dirname(CONFIG["history_save_path"]), exist_ok=True)
    with open(CONFIG["history_save_path"], "w") as f:
        json.dump(combined, f)

    plot_training_history(history, fine_history)
    evaluate_model(model, test_gen, class_names, num_classes)

    print("\n✅ Training complete! Model saved to:", CONFIG["model_save_path"])


if __name__ == "__main__":
    main()
