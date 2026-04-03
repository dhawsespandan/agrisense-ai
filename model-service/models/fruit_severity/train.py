"""
model-service/models/fruit_severity/train.py
============================================
Train EfficientNet-B4 regression head for fruit severity estimation.

Ground-truth severity is computed directly from labelme polygon annotations:

    severity % = affected_pixel_area / total_fruit_pixel_area × 100

Affected area  = union of all disease-label polygons
Total fruit    = union of all polygons (Healthy_Apple + disease)
Healthy images (no JSON) → severity = 0 %

Strategy (CPU-friendly, colour-aware)
--------------------------------------
1. Build EfficientNet-B4 backbone with ImageNet pretrained weights (frozen).
2. For each image extract features N_AUG times, each time under a DIFFERENT
   random colour augmentation (hue/saturation/brightness/contrast jitter +
   random greyscale).  This forces the model to learn severity from shape and
   texture, not from fruit colour — fixing the red-vs-green bias.
3. Cache the augmented feature bank to disk.
4. Train only the linear regression head on the cached features — fast even
   with the larger (augmented) dataset.
5. Use Huber loss (δ=0.1) instead of MSE — more robust to outlier annotations
   from complex/coalesced lesions.
6. Save the full model → weights/fruit_severity_trained.pth

Re-running the script uses the cache when it is valid.

Usage (from repo root):
    python model-service/models/fruit_severity/train.py
"""

import os, sys, glob, json, warnings
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

# ── paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
_MS_ROOT    = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))

DATA_ROOTS = [
    os.path.join(_REPO_ROOT, "green_severity_estimation"),
    os.path.join(_REPO_ROOT, "red_severity_estimation"),
]

SAVE_PATH  = os.path.join(_MS_ROOT, "weights", "fruit_severity_trained.pth")
CACHE_PATH = os.path.join(_MS_ROOT, "weights", ".severity_features_cache_v2.npz")

# ── hyper-parameters ──────────────────────────────────────────────────────────

IMG_SIZE    = 224
BATCH_SIZE  = 32
N_AUG       = 5       # colour-augmented feature extractions per image
HEAD_EPOCHS = 80      # train regression head on cached features
LR          = 5e-4
HUBER_DELTA = 0.1     # Huber loss δ (in [0,1] label space  ≈ 10 % severity)
VAL_SPLIT   = 0.15
DEVICE      = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ── label sets ────────────────────────────────────────────────────────────────

HEALTHY_LABELS = {"Healthy_Apple", "healthy_apple", "Healthy", "healthy"}
IMAGE_EXTS     = (".jpg", ".jpeg", ".png", ".bmp")

# ── severity from polygon annotations ─────────────────────────────────────────

def compute_severity(json_path: str) -> float:
    """
    Rasterise polygon masks from a labelme JSON file and compute:
        severity % = affected_pixels / total_fruit_pixels × 100
    Returns float in [0, 100].
    """
    with open(json_path) as f:
        data = json.load(f)

    w, h = data.get("imageWidth", 512), data.get("imageHeight", 512)

    affected_mask = Image.new("L", (w, h), 0)
    fruit_mask    = Image.new("L", (w, h), 0)
    draw_aff      = ImageDraw.Draw(affected_mask)
    draw_fruit    = ImageDraw.Draw(fruit_mask)

    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        pts   = [tuple(p) for p in shape.get("points", [])]
        if len(pts) < 3:
            continue
        if label in HEALTHY_LABELS:
            draw_fruit.polygon(pts, fill=255)
        else:
            # Disease region: counts toward both affected and total fruit area
            draw_aff.polygon(pts,   fill=255)
            draw_fruit.polygon(pts, fill=255)

    fruit_px    = float(np.array(fruit_mask).sum())    / 255.0
    affected_px = float(np.array(affected_mask).sum()) / 255.0

    if fruit_px < 1.0:
        return 0.0
    return min(100.0, round(affected_px / fruit_px * 100.0, 2))


# ── collect (image_path, severity) pairs ─────────────────────────────────────

def collect_samples(roots):
    samples = []
    for root in roots:
        if not os.path.isdir(root):
            warnings.warn(f"[train] Data root not found: {root}")
            continue
        for disease_dir in sorted(os.listdir(root)):
            full_dir = os.path.join(root, disease_dir)
            if not os.path.isdir(full_dir):
                continue
            for img_file in sorted(os.listdir(full_dir)):
                if not img_file.lower().endswith(IMAGE_EXTS):
                    continue
                img_path  = os.path.join(full_dir, img_file)
                json_path = os.path.splitext(img_path)[0] + ".json"
                severity  = compute_severity(json_path) if os.path.exists(json_path) else 0.0
                samples.append((img_path, severity))
    return samples


# ── colour-augmented transforms ───────────────────────────────────────────────

_BASE_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _augmented_tf(seed: int):
    """
    Return a deterministic colour-augmenting transform.
    Varying hue ±25 % simulates the full red→green apple spectrum.
    """
    rng = random.Random(seed)

    def transform(img: Image.Image) -> torch.Tensor:
        # Geometric
        if rng.random() > 0.5:
            img = TF.hflip(img)
        if rng.random() > 0.5:
            img = TF.vflip(img)
        angle = rng.uniform(-15, 15)
        img = TF.rotate(img, angle)

        # Colour jitter — wide hue range covers red ↔ green ↔ yellow apples
        brightness = rng.uniform(0.6, 1.4)
        contrast   = rng.uniform(0.7, 1.3)
        saturation = rng.uniform(0.5, 1.5)
        hue        = rng.uniform(-0.25, 0.25)   # ±25 % hue shift
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img,   contrast)
        img = TF.adjust_saturation(img, saturation)
        img = TF.adjust_hue(img,        hue)

        # Occasional greyscale (forces shape/texture over colour)
        if rng.random() < 0.15:
            img = TF.to_grayscale(img, num_output_channels=3)

        # Random Gaussian blur to simulate varied lesion edge sharpness
        if rng.random() < 0.3:
            radius = rng.choice([1, 2])
            img = img.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=radius))

        img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        t   = TF.to_tensor(img)
        t   = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return t

    return transform


# ── feature extraction (N_AUG passes per image) ───────────────────────────────

def extract_features(samples, backbone, device):
    """
    Run every image through the frozen backbone N_AUG times, each with a
    different colour augmentation.  Returns:
        features : np.ndarray  (N * N_AUG, feat_dim)
        labels   : np.ndarray  (N * N_AUG,)           [0, 1] normalised
    """
    backbone.eval()
    all_feats  = []
    all_labels = []
    n = len(samples)

    with torch.no_grad():
        for aug_idx in range(N_AUG):
            aug_label = "original" if aug_idx == 0 else f"augment-{aug_idx}"
            print(f"  Pass {aug_idx+1}/{N_AUG}  ({aug_label}) …")

            tf = _BASE_TF if aug_idx == 0 else None   # built per-image below
            batch_imgs, batch_lbls = [], []

            for i, (img_path, severity) in enumerate(samples):
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

                if aug_idx == 0:
                    t = tf(img)
                else:
                    # unique seed per (image, augmentation pass)
                    t = _augmented_tf(seed=aug_idx * 100000 + i)(img)

                batch_imgs.append(t)
                batch_lbls.append(severity / 100.0)

                if len(batch_imgs) == BATCH_SIZE or i == n - 1:
                    tensor = torch.stack(batch_imgs).to(device)
                    feats  = backbone(tensor).cpu().numpy()
                    all_feats.append(feats)
                    all_labels.extend(batch_lbls)
                    batch_imgs, batch_lbls = [], []

            if (aug_idx + 1) % 1 == 0:
                print(f"    {n} images done for pass {aug_idx+1}")

    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    labels   = np.array(all_labels,                  dtype=np.float32)
    return features, labels


# ── feature dataset ───────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── main ──────────────────────────────────────────────────────────────────────

def train():
    print(f"[train] Device  : {DEVICE}")
    print(f"[train] N_AUG   : {N_AUG}  (colour augmentations per image)")
    print(f"[train] Output  : {SAVE_PATH}")

    # 1. Collect samples & compute ground-truth severity labels
    print("\n[1/3] Collecting samples and computing severity labels …")
    samples = collect_samples(DATA_ROOTS)
    if not samples:
        sys.exit("[train] No samples found — check DATA_ROOTS paths.")

    raw_labels = np.array([s[1] for s in samples], dtype=np.float32)
    print(f"      {len(samples)} samples  |  "
          f"mean severity = {raw_labels.mean():.1f}%  |  "
          f"max = {raw_labels.max():.1f}%  |  "
          f"non-zero = {(raw_labels > 0).sum()}")

    # 2. Build frozen backbone
    print("\n[2/3] Building EfficientNet-B4 backbone (pretrained, frozen) …")
    full_model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.DEFAULT
    )
    feat_dim = full_model.classifier[1].in_features   # 1792
    backbone = nn.Sequential(
        full_model.features,
        full_model.avgpool,
        nn.Flatten(),
    ).to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad_(False)

    # 3. Feature extraction / cache
    cache_valid = False
    if os.path.exists(CACHE_PATH):
        print(f"      Found cache at {CACHE_PATH} — validating …")
        try:
            cache = np.load(CACHE_PATH)
            expected_rows = len(samples) * N_AUG
            if (cache["features"].shape == (expected_rows, feat_dim)
                    and cache["n_aug"] == N_AUG
                    and list(cache["paths"]) == [s[0] for s in samples]):
                features, labels = cache["features"], cache["labels"]
                cache_valid = True
                print(f"      Cache valid: {features.shape}")
        except Exception as e:
            print(f"      Cache corrupt ({e}) — re-extracting.")

    if not cache_valid:
        print(f"      Extracting {N_AUG} × {len(samples)} = "
              f"{N_AUG * len(samples)} feature vectors …")
        features, labels = extract_features(samples, backbone, DEVICE)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(
            CACHE_PATH,
            features=features,
            labels=labels,
            paths=np.array([s[0] for s in samples]),
            n_aug=N_AUG,
        )
        print(f"      Saved feature cache → {CACHE_PATH}")

    print(f"      Feature bank: {features.shape}  "
          f"labels min={labels.min():.3f} max={labels.max():.3f}")

    # 4. Train/val split — split on original image index, then expand to augmented copies
    print(f"\n[3/3] Training regression head ({HEAD_EPOCHS} epochs, Huber loss) …")
    n_orig  = len(samples)
    n_val   = max(1, int(n_orig * VAL_SPLIT))
    n_train = n_orig - n_val

    rng   = torch.Generator().manual_seed(42)
    idx   = torch.randperm(n_orig, generator=rng).numpy()
    t_idx, v_idx = idx[:n_train], idx[n_train:]

    # Expand to all augmented copies
    aug_t_idx = np.concatenate([t_idx + aug * n_orig for aug in range(N_AUG)])
    aug_v_idx = v_idx   # validation: original images only (no augmentation leakage)

    train_ds = FeatureDataset(features[aug_t_idx], labels[aug_t_idx])
    val_ds   = FeatureDataset(features[aug_v_idx], labels[aug_v_idx])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

    print(f"      Train vectors: {len(train_ds)}  |  Val images: {len(val_ds)}")

    # Regression head
    head = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    ).to(DEVICE)

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HEAD_EPOCHS, eta_min=1e-5
    )

    best_val_mae = float("inf")

    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        train_loss = 0.0
        for feats, targets in train_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            preds = head(feats)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_ds)

        head.eval()
        val_mae = 0.0
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
                val_mae += (head(feats) - targets).abs().sum().item() * 100.0
        val_mae /= len(val_ds)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{HEAD_EPOCHS}  "
                  f"train_loss={train_loss:.5f}  val_MAE={val_mae:.2f}%")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            # Assemble full model (backbone + trained head) for saving
            full_model.classifier = nn.Sequential(
                full_model.classifier[0],   # original Dropout
                head[0],                    # Linear(1792, 512)
                head[1],                    # ReLU
                head[2],                    # Dropout(0.4)
                head[3],                    # Linear(512, 128)
                head[4],                    # ReLU
                head[5],                    # Dropout(0.2)
                head[6],                    # Linear(128, 1)
                head[7],                    # Sigmoid
            )
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({"model_state": full_model.state_dict()}, SAVE_PATH)

    print(f"\n[train] Done.  Best val MAE = {best_val_mae:.2f}%  →  {SAVE_PATH}")


if __name__ == "__main__":
    train()
