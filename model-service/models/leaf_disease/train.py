"""
models/leaf_disease/train.py
==============================
Training script for Leaf Disease EfficientNet-V2-S

Usage
-----
    # From the model-service/ directory:
    python -m models.leaf_disease.train --pretrained_backbone

    # Point to a custom dataset root (must contain train/ and val/ sub-dirs):
    python -m models.leaf_disease.train \
        --data_dir /path/to/processed_dataset \
        --pretrained_backbone

Dataset layout expected
-----------------------
    <data_dir>/
        train/
            alternaria_leaf_spot/
            brown_spot/
            gray_spot/
            healthy_leaf/
            rust/
        val/   (same structure)
        test/  (optional, same structure)

Two-phase training strategy
----------------------------
Phase 1 — Head warm-up  (backbone fully frozen)
    Epochs   : 100   (--phase1_epochs)
    LR       : 1e-3
    Patience : 25    (--phase1_patience, monitors val macro-F1)

Phase 2 — Gradual fine-tune  (top backbone blocks unlocked incrementally)
    2a — top 2 EfficientNet feature blocks + classifier (LR 1e-5)
    2b — all layers (LR 1e-5, continued from 2a checkpoint)
    Epochs   : 50   per sub-phase (--phase2_epochs)
    Patience : 15   per sub-phase (--phase2_patience)

Class-imbalance handling
-------------------------
WeightedRandomSampler  — balances per-batch class frequency.
CrossEntropyLoss(weight=…, label_smoothing=0.1)
    — inverse-frequency class weights further penalise majority errors.

Best model selection
---------------------
Checkpoint is saved only when validation MACRO-F1 improves (not accuracy),
ensuring balanced performance across all 5 classes including minority ones.

Outputs
-------
    <output_path>               best model checkpoint (.pt)
    <output_dir>/training_log.csv
    <output_dir>/confusion_matrix_val.png
    <output_dir>/confusion_matrix_test.png  (if test/ exists)

After training
--------------
Set LEAF_MODEL_PATH to your checkpoint and restart the service:
    export LEAF_MODEL_PATH=<output_path>
"""

import argparse
import csv
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CLASSES = 5
_MEAN       = [0.485, 0.456, 0.406]
_STD        = [0.229, 0.224, 0.225]

DEFAULT_DATA = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",          # → project root
    "processed_dataset",
))
DEFAULT_OUT = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..",                # → model-service/
    "weights", "leaf_efficientnetv2s.pt",
))


# ── Transforms ────────────────────────────────────────────────────────────
# ColorJitter covers lighting/colour variance from mobile cameras.
# RandomGrayscale forces the model to learn texture, not just colour hue.
# Full ImageNet normalisation must be kept in sync with model.py.

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ── Architecture ──────────────────────────────────────────────────────────

def build_leaf_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    EfficientNet-V2-S with a two-layer classification head.
    Deeper head gives better separation for visually similar classes
    (Brown Spot vs Rust vs Alternaria) on small datasets.
    This function is the single source of truth — model.py mirrors it exactly.
    """
    backbone    = tv_models.efficientnet_v2_s(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(256, num_classes),
    )
    return backbone


# ── Data helpers ──────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    WeightedRandomSampler so that every class is seen equally per epoch,
    regardless of how many source images it has.
    Critical for minority classes (brown_spot, alternaria_leaf_spot).
    """
    counts        = np.bincount(dataset.targets)
    class_weights = 1.0 / counts.astype(float)
    sample_w      = class_weights[dataset.targets]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_w).float(),
        num_samples=len(sample_w),
        replacement=True,
    )


def make_loss_weights(dataset: datasets.ImageFolder, device: str) -> torch.Tensor:
    """
    Inverse-frequency per-class loss weights for CrossEntropyLoss.
    Doubles down on class-imbalance correction alongside the sampler.
    """
    counts  = np.bincount(dataset.targets).astype(float)
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= len(counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── Phase helpers ─────────────────────────────────────────────────────────

def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the classifier head."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier")


def unfreeze_top_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    """
    Unfreeze only the last n_blocks of model.features plus the classifier.
    Used for Phase 2a gradual unfreezing — lets the backbone adapt gently
    at low LR before opening up all layers, reducing catastrophic forgetting.
    EfficientNet-V2-S has 8 feature sub-modules (indices 0-7).
    """
    for p in model.parameters():
        p.requires_grad = False
    feature_blocks = list(model.features.children())
    total = len(feature_blocks)
    for i, block in enumerate(feature_blocks):
        if i >= total - n_blocks:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


# ── Evaluation & reporting ─────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], save_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title("Leaf Disease — Confusion Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {save_path}")


# ── Train / eval loops ────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Single training epoch.
    Fix: logits are saved from the first forward pass and reused for both
    loss and accuracy — eliminates the previous double-forward-pass bug
    that computed accuracy on post-step weights and wasted 2x compute.
    """
    model.train()
    running_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct      += (logits.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, class_names):
    """
    Full validation pass.
    Returns (loss, accuracy, macro_f1, confusion_matrix).
    Prints per-class Precision / Recall / F1 (Classification Report) so
    misclassifications between Brown Spot / Rust / Alternaria are visible.
    """
    model.eval()
    running_loss = correct = total = 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        running_loss += criterion(logits, labels).item() * imgs.size(0)
        preds   = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)
    print(classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))
    return running_loss / total, correct / total, macro_f1, cm


# ── Early stopping ────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience    = patience
        self.best        = -1.0
        self.counter     = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if metric > self.best:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Checkpoint helper ─────────────────────────────────────────────────────

def _save_checkpoint(model, out_path, val_acc, val_f1, best_f1, best_cm, cm):
    if val_f1 > best_f1:
        torch.save({
            "model_state": model.state_dict(),
            "val_acc":     val_acc,
            "val_f1":      val_f1,
            "head":        "two_layer",
            "normalised":  True,
        }, str(out_path))
        print(f"  ✓ Checkpoint saved (F1={val_f1:.4f})")
        return val_f1, cm
    return best_f1, best_cm


# ── Main ──────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected train/ and val/ inside '{args.data_dir}'.\n"
            "Either run prepare_data.py first or pass --data_dir pointing to the "
            "directory that already has train/ and val/ sub-folders."
        )

    train_ds = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    val_ds   = datasets.ImageFolder(val_dir,   transform=EVAL_TRANSFORM)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"Classes (sorted): {train_ds.classes}\n")
    print(f"Class counts (train): { {c: sum(1 for t in train_ds.targets if t == i) for i, c in enumerate(train_ds.classes)} }\n")

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    loss_weights = make_loss_weights(train_ds, device)
    criterion    = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)

    model = build_leaf_model(NUM_CLASSES)
    if args.pretrained_backbone:
        print("Loading ImageNet pretrained backbone …")
        src  = tv_models.efficientnet_v2_s(
            weights=tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        miss, _ = model.load_state_dict(src.state_dict(), strict=False)
        print(f"  Loaded (missing head keys as expected: {len(miss)})")
    model.to(device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "training_log.csv"

    best_f1  = -1.0
    best_cm  = None
    log_rows = []

    # ── Phase 1 — Head warm-up (backbone fully frozen) ───────────────────
    print("=" * 66)
    print(f"Phase 1 — Head warm-up  "
          f"(frozen backbone | LR=1e-3 | {args.phase1_epochs} epochs | patience={args.phase1_patience})")
    print("=" * 66)
    freeze_backbone(model)
    opt1   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.phase1_epochs)
    es1    = EarlyStopping(patience=args.phase1_patience)

    for ep in range(1, args.phase1_epochs + 1):
        tr_loss, tr_acc            = train_epoch(model, train_loader, criterion, opt1, device)
        vl_loss, vl_acc, vl_f1, cm = eval_epoch(model, val_loader, criterion, device, train_ds.classes)
        sched1.step()
        print(f"[P1 {ep:03d}/{args.phase1_epochs}] "
              f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} val_f1={vl_f1:.4f}")
        log_rows.append(dict(phase=1, epoch=ep,
                             train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                             val_loss=round(vl_loss, 4),   val_acc=round(vl_acc, 4),
                             val_f1=round(vl_f1, 4)))
        best_f1, best_cm = _save_checkpoint(
            model, out_path, vl_acc, vl_f1, best_f1, best_cm, cm
        )
        if es1.step(vl_f1):
            print(f"  Early stopping at epoch {ep} (patience={args.phase1_patience})")
            break

    # ── Phase 2a — Gradual fine-tune (top 2 feature blocks) ──────────────
    print("=" * 66)
    print(f"Phase 2a — Gradual fine-tune  "
          f"(top 2 blocks + head | LR=1e-5 | {args.phase2_epochs} epochs | patience={args.phase2_patience})")
    print("=" * 66)
    unfreeze_top_blocks(model, n_blocks=2)
    opt2a   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5, weight_decay=1e-4,
    )
    sched2a = torch.optim.lr_scheduler.CosineAnnealingLR(opt2a, T_max=args.phase2_epochs)
    es2a    = EarlyStopping(patience=args.phase2_patience)

    for ep in range(1, args.phase2_epochs + 1):
        tr_loss, tr_acc            = train_epoch(model, train_loader, criterion, opt2a, device)
        vl_loss, vl_acc, vl_f1, cm = eval_epoch(model, val_loader, criterion, device, train_ds.classes)
        sched2a.step()
        print(f"[P2a {ep:03d}/{args.phase2_epochs}] "
              f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} val_f1={vl_f1:.4f}")
        log_rows.append(dict(phase="2a", epoch=ep,
                             train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                             val_loss=round(vl_loss, 4),   val_acc=round(vl_acc, 4),
                             val_f1=round(vl_f1, 4)))
        best_f1, best_cm = _save_checkpoint(
            model, out_path, vl_acc, vl_f1, best_f1, best_cm, cm
        )
        if es2a.step(vl_f1):
            print(f"  Early stopping at epoch {ep} (patience={args.phase2_patience})")
            break

    # ── Phase 2b — Full fine-tune (all layers) ────────────────────────────
    print("=" * 66)
    print(f"Phase 2b — Full fine-tune  "
          f"(all layers | LR=1e-5 | {args.phase2_epochs} epochs | patience={args.phase2_patience})")
    print("=" * 66)
    unfreeze_all(model)
    opt2b   = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    sched2b = torch.optim.lr_scheduler.CosineAnnealingLR(opt2b, T_max=args.phase2_epochs)
    es2b    = EarlyStopping(patience=args.phase2_patience)

    for ep in range(1, args.phase2_epochs + 1):
        tr_loss, tr_acc            = train_epoch(model, train_loader, criterion, opt2b, device)
        vl_loss, vl_acc, vl_f1, cm = eval_epoch(model, val_loader, criterion, device, train_ds.classes)
        sched2b.step()
        print(f"[P2b {ep:03d}/{args.phase2_epochs}] "
              f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} val_f1={vl_f1:.4f}")
        log_rows.append(dict(phase="2b", epoch=ep,
                             train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                             val_loss=round(vl_loss, 4),   val_acc=round(vl_acc, 4),
                             val_f1=round(vl_f1, 4)))
        best_f1, best_cm = _save_checkpoint(
            model, out_path, vl_acc, vl_f1, best_f1, best_cm, cm
        )
        if es2b.step(vl_f1):
            print(f"  Early stopping at epoch {ep} (patience={args.phase2_patience})")
            break

    # ── Reporting ─────────────────────────────────────────────────────────
    print(f"\nTraining complete. Best val F1 = {best_f1:.4f}")
    print(f"Checkpoint → {out_path}")

    if best_cm is not None:
        plot_confusion_matrix(
            best_cm, train_ds.classes,
            str(out_path.parent / "confusion_matrix_val.png"),
        )

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["phase", "epoch", "train_loss", "train_acc",
                           "val_loss", "val_acc", "val_f1"]
        )
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Training log → {log_path}")

    # ── Optional test-set evaluation ──────────────────────────────────────
    test_dir = os.path.join(args.data_dir, "test")
    if os.path.isdir(test_dir):
        print("\n── Test-set evaluation ──────────────────────────────────")
        test_ds  = datasets.ImageFolder(test_dir, transform=EVAL_TRANSFORM)
        test_ldr = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
        ckpt = torch.load(str(out_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        _, test_acc, test_f1, test_cm = eval_epoch(
            model, test_ldr, criterion, device, test_ds.classes
        )
        print(f"Test acc={test_acc:.4f}  F1={test_f1:.4f}")
        plot_confusion_matrix(
            test_cm, test_ds.classes,
            str(out_path.parent / "confusion_matrix_test.png"),
        )

    print("\n── Service integration reminder ─────────────────────────────────")
    print(f"  Set LEAF_MODEL_PATH={out_path} before starting model-service.")
    print("  model.py already uses the two-layer head and ImageNet normalisation")
    print("  — no further changes required after retraining.")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-V2-S for Apple Leaf Disease Detection"
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.normpath(DEFAULT_DATA),
        help="Directory containing train/ and val/ sub-folders "
             "(default: <project_root>/processed_dataset)",
    )
    parser.add_argument(
        "--output",
        default=os.path.normpath(DEFAULT_OUT),
        help="Path to save the best checkpoint (.pt)",
    )
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument(
        "--phase1_epochs",   type=int, default=100,
        help="Head warm-up epochs (backbone frozen, LR=1e-3)",
    )
    parser.add_argument(
        "--phase1_patience", type=int, default=25,
        help="Early stopping patience for Phase 1",
    )
    parser.add_argument(
        "--phase2_epochs",   type=int, default=50,
        help="Epochs per gradual fine-tune sub-phase (2a and 2b, LR=1e-5)",
    )
    parser.add_argument(
        "--phase2_patience", type=int, default=15,
        help="Early stopping patience per Phase 2 sub-phase",
    )
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--device",          default=None)
    parser.add_argument(
        "--pretrained_backbone", action="store_true",
        help="Load ImageNet backbone weights before training (strongly recommended)",
    )
    args = parser.parse_args()
    main(args)
