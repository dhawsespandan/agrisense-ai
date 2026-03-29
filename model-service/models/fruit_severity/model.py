"""
models/fruit_severity/model.py
================================
Fruit Disease Severity Estimation — EfficientNet-B4 regression (ViT-style)

Outputs a severity score in the range [0, 100] %.

If the weights file is missing (development mode) the model is initialised
with random weights and a warning is printed.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.serialization
import torchvision.models as models

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

DEVICE     = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_CKPT_PATH = os.getenv("SEVERITY_MODEL_PATH", "weights/efficientnetb4_spandan.pth")

# ── load checkpoint or fall back to random weights ────────────────────────

_model = models.efficientnet_b4(weights=None)
_model.classifier[1] = nn.Linear(_model.classifier[1].in_features, 1)

if os.path.exists(_CKPT_PATH):
    _ckpt  = torch.load(_CKPT_PATH, map_location=DEVICE, weights_only=False)
    _state = {}
    for k, v in _ckpt["model_state"].items():
        new_k = k.replace("backbone.", "features.", 1) if k.startswith("backbone.") else k
        _state[new_k] = v
    _model.load_state_dict(_state, strict=False)
    print("FruitSeverity [EfficientNet-B4] loaded — output: severity % (0–100)")
else:
    warnings.warn(
        f"[fruit_severity] Weights not found at '{_CKPT_PATH}'. "
        "Using random EfficientNet-B4 weights — severity scores are meaningless "
        "until real weights are supplied.",
        RuntimeWarning, stacklevel=2,
    )

_model.to(DEVICE).eval()


# ── public API ────────────────────────────────────────────────────────────

def predict_severity(tensor: torch.Tensor) -> float:
    """
    Parameters
    ----------
    tensor : torch.Tensor — (1, 3, H, W) already on DEVICE

    Returns
    -------
    float — severity percentage clamped to [0, 100]
    """
    with torch.no_grad():
        raw = _model(tensor).item() * 100
    return round(max(0.0, min(100.0, raw)), 2)
