# AgriSense AI — Apple Orchard Health Monitoring

## Project Overview

A full-stack AI-powered apple crop disease detection app. Users upload images of apple leaves, fruit, or flower clusters, and the system classifies the disease, estimates severity, and provides actionable recommendations.

## Architecture

Three-tier application:

1. **client/** — React + Vite + TypeScript + Tailwind CSS frontend (port 5000)
2. **server/** — Node.js/Express backend API gateway (port 3001)
3. **model-service/** — Python FastAPI ML inference service (port 8000)

### Request Flow
```
Browser → Vite (5000) → [proxy /api] → Express (3001) → FastAPI (8000)
```

## ML Pipeline

The FastAPI service runs a multi-stage pipeline:
- **Router**: EfficientNet-B0 (with DINOv2+LoRA fallback) classifies image as fruit/leaf/flower
- **Fruit branch**: CLIP apple detector → EfficientNet-B2 disease classifier → EfficientNet-B4 severity estimator
- **Leaf branch**: CLIP leaf detector → EfficientNet-V2-S disease classifier (5 classes)
- **Flower branch**: YOLOv8-m cluster detector → health label

## Model Weights

Located in `model-service/weights/`:
- `router_efficientnet.pt` — image type router
- `router_model_final/` — DINOv2+LoRA router (optional)
- `fruit_efficientnet.pt` — fruit disease classifier
- `fruit_severity_trained.pth` / `efficientnetb4_spandan.pth` — severity estimator
- `leaf_disease_v2.pt` / `efficientnetb0_astha.pt` — leaf disease classifier
- `yolo26m_abhirami.pt` — flower cluster YOLO detector

## Environment Variables

**server/.env**
```
PORT=3001
PYTHON_SERVICE_URL=http://localhost:8000
```

**model-service/.env**
```
DEVICE=cpu
ROUTER_LORA_DIR=weights/router_model_final
ROUTER_EFFICIENTNET_PATH=weights/router_efficientnet.pt
FRUIT_MODEL_PATH=weights/fruit_efficientnet.pt
SEVERITY_MODEL_PATH=weights/efficientnetb4_spandan.pth
LEAF_MODEL_PATH=weights/efficientnetb0_astha.pt
FLOWER_MODEL_PATH=weights/yolo26m_abhirami.pt
```

## Workflows

- **Start application** — `cd client && npm run dev` (port 5000, webview)
- **Backend Server** — `cd server && node app.js` (port 3001, console)
- **Model Service** — `cd model-service && uvicorn main:app --host 127.0.0.1 --port 8000` (console, no waitForPort — ML models take ~30-60s to load)

## Important Notes

- Model service takes 30-60 seconds to start due to ML model loading. The workflow system may report it as "failed" but it continues running.
- The model service workflow intentionally has no `waitForPort` setting to avoid premature termination during model loading.
- System dependency `xorg.libxcb` is required for OpenCV/YOLO (installed via Nix).
- Frontend proxies `/api` requests to the Express backend (configured in `client/vite.config.ts`).
- `DEVICE=cpu` is set in model-service — no GPU required.

## Dependencies

- **client**: React 18, Vite 6, TypeScript, Tailwind CSS v4
- **server**: Express, multer, axios, cors, dotenv, form-data
- **model-service**: FastAPI, uvicorn, PyTorch (CPU), torchvision, transformers, peft, ultralytics
