"""
models/leaf_disease/model.py
==============================
Leaf Disease Detection CNN — EfficientNet-V2-S

Classifies an apple leaf image into one of 5 classes:
    alternaria leaf spot | brown spot | gray spot | healthy leaf | rust

If the weights file is missing (development mode) the model is initialised
with random weights and a warning is printed.
"""

import os
import warnings
import torch
import torch.nn as nn
import torchvision.models as tv_models

DEVICE     = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_CKPT_PATH = os.getenv("LEAF_MODEL_PATH", "weights/efficientnetb0_astha.pt")

LEAF_CLASSES = [
    "alternaria leaf spot",
    "brown spot",
    "gray spot",
    "healthy leaf",
    "rust",
]

# ── load checkpoint or fall back to random weights ────────────────────────

_model = tv_models.efficientnet_v2_s(weights=None)
_model.classifier[1] = nn.Linear(
    _model.classifier[1].in_features, len(LEAF_CLASSES)
)

if os.path.exists(_CKPT_PATH):
    _ckpt = torch.load(_CKPT_PATH, map_location=DEVICE)
    if "model_state" in _ckpt:
        _state = _ckpt["model_state"]
    elif "state_dict" in _ckpt:
        _state = _ckpt["state_dict"]
    elif "model" in _ckpt:
        _state = _ckpt["model"]
    else:
        _state = _ckpt
    _model.load_state_dict(_state, strict=False)
    print(f"LeafDisease [EfficientNet-V2-S] loaded — classes: {LEAF_CLASSES}")
else:
    warnings.warn(
        f"[leaf_disease] Weights not found at '{_CKPT_PATH}'. "
        "Using random EfficientNet-V2-S weights — predictions are meaningless "
        "until real weights are supplied.",
        RuntimeWarning, stacklevel=2,
    )

_model.to(DEVICE).eval()


# ── public API ────────────────────────────────────────────────────────────

def predict_leaf(tensor: torch.Tensor) -> tuple[str, float]:
    """
    Parameters
    ----------
    tensor : torch.Tensor — (1, 3, H, W) already on DEVICE

    Returns
    -------
    (label, confidence)  where label ∈ LEAF_CLASSES and confidence ∈ [0, 1]
    """
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return LEAF_CLASSES[idx], round(probs[idx].item(), 4)
