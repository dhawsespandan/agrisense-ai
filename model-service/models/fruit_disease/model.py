"""
models/fruit_disease/model.py
==============================
Fruit Disease Detection CNN — EfficientNet-B2

Classifies an apple fruit image into one of 5 classes:
    Anthracnose | Blotch | Healthy | Rot | Scab

If the weights file is missing (development mode) the model is initialised
with random weights and a warning is printed — the API will still respond
with mock-shaped output so the pipeline can be tested end-to-end.
"""

import os
import warnings
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE     = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_CKPT_PATH = os.getenv("FRUIT_MODEL_PATH", "weights/fruit_efficientnet.pt")

DEFAULT_CLASSES = ["Anthracnose", "Blotch", "Healthy", "Rot", "Scab"]

# ── load checkpoint or fall back to random weights ────────────────────────

if os.path.exists(_CKPT_PATH):
    _ckpt    = torch.load(_CKPT_PATH, map_location=DEVICE)
    _CLASSES = _ckpt["classes"]
    _model   = models.efficientnet_b2(weights=None)
    _model.classifier[1] = nn.Linear(
        _model.classifier[1].in_features, len(_CLASSES)
    )
    _model.load_state_dict(_ckpt["model_state"])
    print(f"FruitDisease [EfficientNet-B2] loaded — "
          f"classes: {_CLASSES}  val_acc: {_ckpt.get('val_acc', '?'):.4f}")
else:
    warnings.warn(
        f"[fruit_disease] Weights not found at '{_CKPT_PATH}'. "
        "Using random EfficientNet-B2 weights — predictions are meaningless "
        "until real weights are supplied.",
        RuntimeWarning, stacklevel=2,
    )
    _CLASSES = DEFAULT_CLASSES
    _model   = models.efficientnet_b2(weights=None)
    _model.classifier[1] = nn.Linear(
        _model.classifier[1].in_features, len(_CLASSES)
    )

_model.to(DEVICE).eval()

# ── pre-process ───────────────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── public API ────────────────────────────────────────────────────────────

def predict_fruit(image_path: str) -> tuple[str, float]:
    """
    Parameters
    ----------
    image_path : str — path to the image on disk

    Returns
    -------
    (label, confidence)  where label ∈ _CLASSES and confidence ∈ [0, 1]
    """
    img    = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()

    return _CLASSES[idx], round(probs[idx].item(), 4)
