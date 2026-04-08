---
title: AgriSense AI Model Service
emoji: 🍎
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# AgriSense AI — Model Service

FastAPI inference service for apple crop disease detection.

## Endpoints

- `GET /health` — Health check
- `GET /docs` — Interactive API docs (Swagger UI)
- `POST /predict` — Upload an image and receive a disease prediction

## Pipeline

1. CLIP zero-shot pre-filter (rejects non-apple images)
2. Router (DINOv2+LoRA or EfficientNet-B0 fallback)
3. Specialised branch: Fruit / Leaf / Flower
