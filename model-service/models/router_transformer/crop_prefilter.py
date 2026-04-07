"""
models/router_transformer/crop_prefilter.py
=============================================
Gate-0: Apple Crop / Not-Apple-Crop binary check using CLIP zero-shot classification.

Runs BEFORE the router so that images that are clearly not apple leaves,
fruits, or flower clusters are rejected immediately — without even reaching
the EfficientNet/LoRA router, which was trained only on apple crop imagery
and will blindly assign one of the three crop classes to anything.

Pipeline
--------
   image  →  CLIP  →  P("apple leaf, fruit or flower")
                       P("not an apple crop")
           ≥ threshold  →  True (pass through)
           < threshold  →  False (reject with 'unknown')

The CLIP model (openai/clip-vit-base-patch32) is lazy-loaded on first call
and shared with apple_detector.py if that module is already imported.

Environment variables
---------------------
CROP_PREFILTER_THRESHOLD : float  (default 0.50)
    Minimum CLIP probability for the apple-crop label to let the image
    through.  0.50 is deliberately lenient so borderline close-ups
    (e.g. a single flower petal or small fruit section) still pass.
    Raise it to be stricter; lower it only if real crop images are
    being rejected.
"""

import os
import warnings
import torch
from PIL import Image

DEVICE = os.getenv("DEVICE", "cpu")
_HF_DEVICE = 0 if DEVICE == "cuda" else -1

CROP_THRESHOLD = float(os.getenv("CROP_PREFILTER_THRESHOLD", "0.50"))

# Candidate labels. Keep them concise — CLIP performs best with short,
# descriptive noun phrases.
_LABELS = [
    "apple leaf, fruit or flower cluster",
    "not an apple crop image",
]
_CROP_LABEL = "apple leaf, fruit or flower cluster"

_pipeline = None


def _load_pipeline() -> None:
    global _pipeline
    if _pipeline is not None:
        return
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=_HF_DEVICE,
        )
        print("[crop_prefilter] CLIP model loaded (openai/clip-vit-base-patch32)")
    except Exception as exc:
        warnings.warn(
            f"[crop_prefilter] Could not load CLIP model: {exc}\n"
            "Crop pre-filtering will be skipped — all images will be forwarded "
            "to the router (fail-open behaviour).",
            RuntimeWarning,
            stacklevel=2,
        )
        _pipeline = None


def is_apple_crop(image: Image.Image) -> tuple[bool, float]:
    """
    Check whether *image* contains apple leaf, fruit, or flower content.

    Parameters
    ----------
    image : PIL.Image.Image
        Already-opened RGB image.

    Returns
    -------
    (is_crop, confidence)
        is_crop    – True if CLIP believes this is an apple crop image.
        confidence – CLIP score for the positive label ∈ [0, 1].
                     Returns -1.0 and True if CLIP failed to load
                     (fail-open so the rest of the pipeline still works).
    """
    _load_pipeline()

    if _pipeline is None:
        return True, -1.0

    img_rgb = image.convert("RGB")
    results = _pipeline(img_rgb, candidate_labels=_LABELS)

    crop_score = next(
        (r["score"] for r in results if r["label"] == _CROP_LABEL), 0.0
    )

    return crop_score >= CROP_THRESHOLD, round(crop_score, 4)
