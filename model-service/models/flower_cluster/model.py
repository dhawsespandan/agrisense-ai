"""
models/flower_cluster/model.py
================================
Flower Cluster Detection CNN — YOLOv8-m

Detects flower clusters in an apple-tree image and returns a health label:
    Healthy Clusters  — > 5 detections (abundant)
    Sparse Clusters   — 1–5 detections (sparse)
    No Clusters Detected — 0 detections

If the weights file is missing (development mode) a mock predictor is used
that always returns "Sparse Clusters" with a confidence of 0.5 and a
warning is printed.
"""

import os
import warnings

_MODEL_PATH = os.getenv("FLOWER_MODEL_PATH", "weights/yolo26m_abhirami.pt")

# ── load YOLO or create mock fallback ────────────────────────────────────

_yolo = None

if os.path.exists(_MODEL_PATH):
    try:
        from ultralytics import YOLO
        _yolo = YOLO(_MODEL_PATH)
        print(f"FlowerCluster [YOLOv8-m] loaded — weight: {_MODEL_PATH}")
    except Exception as exc:
        warnings.warn(
            f"[flower_cluster] YOLO load failed ({exc}). "
            "Using mock predictor — predictions are meaningless until "
            "real weights are supplied.",
            RuntimeWarning, stacklevel=2,
        )
else:
    warnings.warn(
        f"[flower_cluster] Weights not found at '{_MODEL_PATH}'. "
        "Using mock predictor — predictions are meaningless until "
        "real weights are supplied.",
        RuntimeWarning, stacklevel=2,
    )


def _count_to_health(count: int) -> tuple[str, float]:
    """Map raw detection count → (health_label, pseudo_confidence)."""
    if count > 5:
        return "Healthy Clusters", 0.92
    elif count >= 1:
        return "Sparse Clusters", 0.80
    else:
        return "No Clusters Detected", 0.88


# ── public API ────────────────────────────────────────────────────────────

def predict_flowers(image_path: str) -> tuple[str, float]:
    """
    Parameters
    ----------
    image_path : str — path to the image on disk

    Returns
    -------
    (health_label, confidence)
        health_label : 'Healthy Clusters' | 'Sparse Clusters' | 'No Clusters Detected'
        confidence   : float in [0, 1]
    """
    if _yolo is None:
        # mock fallback for development
        return "Sparse Clusters", 0.50

    results = _yolo(image_path)
    count   = int(len(results[0].boxes))
    return _count_to_health(count)
