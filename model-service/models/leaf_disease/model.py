"""
models/leaf_disease/model.py
==============================
Leaf Disease Detection — EfficientNet-V2-S  (inference only)

Classifies an apple leaf image into one of 5 classes:
    alternaria leaf spot | brown spot | gray spot | healthy leaf | rust

Architecture
------------
Standard EfficientNet-V2-S with the default single-layer classification head:
    nn.Dropout(p=0.2) → nn.Linear(1280, num_classes)
This matches the efficientnetv2s_astha.pt checkpoint exactly.

Preprocessing
-------------
This module owns its own transform (ImageNet normalisation included) so that
main.py does not need to manage preprocessing for the leaf branch.
predict_leaf() accepts a PIL Image directly.
"""

import os
import warnings

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image

DEVICE     = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_CKPT_PATH = os.getenv("LEAF_MODEL_PATH", "weights/efficientnetv2s_astha.pt")

LEAF_CLASSES = [
    "alternaria leaf spot",
    "brown spot",
    "gray spot",
    "healthy leaf",
    "rust",
]
NUM_CLASSES = len(LEAF_CLASSES)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_LEAF_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


# ── Architecture — must exactly match the checkpoint ───────────────────────

def _build_model() -> nn.Module:
    """
    Standard EfficientNet-V2-S with the default single-layer head.
    classifier.0 = Dropout(0.2)
    classifier.1 = Linear(1280, num_classes)
    This is the architecture used to produce efficientnetv2s_astha.pt.
    """
    model = tv_models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


# ── Load weights ────────────────────────────────────────────────────────────

_model = _build_model()

if os.path.exists(_CKPT_PATH):
    raw = torch.load(_CKPT_PATH, map_location=DEVICE, weights_only=False)

    # Support both bare state-dicts and wrapped checkpoints
    if isinstance(raw, dict) and any(
        k in raw for k in ("model_state", "state_dict", "model")
    ):
        state = (
            raw.get("model_state")
            or raw.get("state_dict")
            or raw.get("model")
        )
    else:
        state = raw  # bare state dict (efficientnetv2s_astha.pt is this format)

    missing, unexpected = _model.load_state_dict(state, strict=True)
    if missing:
        warnings.warn(
            f"[leaf_disease] Missing keys in checkpoint: {missing[:4]}",
            RuntimeWarning, stacklevel=2,
        )
    if unexpected:
        warnings.warn(
            f"[leaf_disease] Unexpected keys in checkpoint: {unexpected[:4]}",
            RuntimeWarning, stacklevel=2,
        )

    val_acc = raw.get("val_acc", "n/a") if isinstance(raw, dict) else "n/a"
    print(
        f"LeafDisease [EfficientNet-V2-S] loaded — "
        f"classes: {LEAF_CLASSES}  val_acc: {val_acc}"
    )
else:
    warnings.warn(
        f"[leaf_disease] Weights not found at '{_CKPT_PATH}'. "
        "Predictions are meaningless until real weights are supplied.",
        RuntimeWarning, stacklevel=2,
    )

_model.to(DEVICE).eval()


# ── Public inference API ────────────────────────────────────────────────────

def predict_leaf(image: "Image.Image | torch.Tensor") -> tuple[str, float]:
    """
    Run leaf disease inference.

    Parameters
    ----------
    image : PIL.Image.Image  — preferred; preprocessing is applied internally.
            torch.Tensor (1, 3, 224, 224) — legacy path; tensor used as-is.

    Returns
    -------
    (label, confidence)
        label      — one of LEAF_CLASSES
        confidence — softmax probability in [0, 1], rounded to 4 d.p.
    """
    if isinstance(image, Image.Image):
        tensor = _LEAF_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    else:
        tensor = image.to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    idx = probs.argmax().item()
    return LEAF_CLASSES[idx], round(probs[idx].item(), 4)


def predict_leaf_topk(
    image: "Image.Image | torch.Tensor", k: int = 3
) -> list[tuple[str, float]]:
    """
    Return the top-k predictions with probabilities.
    Useful for debugging confusion between similar classes.
    """
    if isinstance(image, Image.Image):
        tensor = _LEAF_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    else:
        tensor = image.to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]

    top_probs, top_idx = torch.topk(probs, k)
    return [
        (LEAF_CLASSES[i.item()], round(p.item(), 4))
        for p, i in zip(top_probs, top_idx)
    ]
