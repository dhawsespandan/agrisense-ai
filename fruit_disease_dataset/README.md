# 🍎 Apple Disease Detection — EfficientNet-B0

Detects **5 disease categories** in mature red apple images:

| Class | Description |
|---|---|
| Anthracnose | Dark circular spots with rotting symptoms |
| Black Pox | Dark, corky or scabby patches |
| Black Rot | Dark, sunken lesions |
| Powdery Mildew | White powder-like fungal growth |
| Healthy | No visible disease |

---

## 📁 Project Structure

```
apple_disease_detection/
├── prepare_dataset.py    # Step 1 — organise annotated images into train/val/test
├── train.py              # Step 2 — train EfficientNet-B0 model
├── predict.py            # Step 3 — run inference on new images
├── gradcam.py            # Step 4 (optional) — visualise what the model sees
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Put all your annotated images and X-AnyLabeling JSON files in a folder called `raw_data/`:
```
raw_data/
    img001.jpg
    img001.json
    img002.jpg
    img002.json
    ...
```
Then run:
```bash
python prepare_dataset.py
```
This creates `dataset/train/`, `dataset/val/`, `dataset/test/` with class subfolders.

**If you already have your dataset in class folders**, skip this step and make sure your structure looks like:
```
dataset/
    train/
        Anthracnose/   *.jpg ...
        Black_Pox/
        Black_Rot/
        Healthy/
        Powdery_Mildew/
    val/
        ...
    test/
        ...
```

### 3. Train the model
```bash
python train.py
```

Training runs in **two phases**:
- **Phase 1 (warm-up)**: base EfficientNet-B0 frozen, only classification head trained
- **Phase 2 (fine-tuning)**: top layers of EfficientNet-B0 unfrozen and trained with a lower learning rate

Model is saved to `models/apple_disease_efficientnetb0.h5`.  
Plots are saved to `results/`.

### 4. Predict on new images
```bash
# Single image
python predict.py --image path/to/apple.jpg

# All images in a folder
python predict.py --folder path/to/my_test_images/

# Save output image
python predict.py --image apple.jpg --save results/my_result.png
```

### 5. Explain predictions with Grad-CAM (optional)
```bash
python gradcam.py --image path/to/apple.jpg --save results/gradcam.png
```
Shows which region of the apple the model focused on to make its decision.

---

## ⚙️ Configuration

All hyperparameters are in the `CONFIG` dict at the top of `train.py`:

| Parameter | Default | Description |
|---|---|---|
| `img_size` | (224, 224) | Input size for EfficientNet-B0 |
| `batch_size` | 32 | Batch size |
| `epochs` | 50 | Warm-up training epochs |
| `learning_rate` | 1e-4 | Initial learning rate |
| `fine_tune_lr` | 1e-5 | Fine-tuning learning rate |
| `fine_tune_at` | 100 | Unfreeze layers from this index onward |
| `fine_tune_epochs` | 20 | Fine-tuning epochs |
| `data_dir` | "dataset" | Root dataset folder |

---

## 📊 Expected Output Files

After training:
```
models/
    apple_disease_efficientnetb0.h5   ← best model checkpoint
    training_history.json
results/
    training_curves.png               ← accuracy / loss plots
    confusion_matrix.png              ← class-level performance
```

---

## 📌 Notes on Your Workflow

- **Layer-cake annotations**: Each image has a `healthy` label (whole apple) and optionally a disease label (the lesion region). `prepare_dataset.py` automatically reads your JSON files and classifies the image by the disease label if present, otherwise as Healthy.
- **Adding more diseases**: To add Black Pox, Black Rot, or Powdery Mildew later, annotate those images in X-AnyLabeling, put the JSONs in `raw_data/`, and re-run `prepare_dataset.py`. The label mapping is in the `DISEASE_LABELS` dict at the top of that file.
- **GPU**: Training is much faster on a CUDA-enabled GPU. TensorFlow will use it automatically if available.
