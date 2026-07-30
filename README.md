# PlantCare AI

An AI-powered web application for plant disease classification using deep learning and computer vision. Users can upload a leaf image to receive a disease prediction with a confidence score, along with detailed information about the disease including symptoms, causes, treatment, and prevention. Every prediction is stored in a SQLite database for future reference.

## Highlights

- ResNet50 transfer-learning model, 98% validation accuracy across 15 plant disease classes
- Full pipeline: PyTorch training/eval → FastAPI backend → React frontend → SQLite history → Docker
- Shared `model_factory.py` between training and inference — serving model always matches the trained checkpoint's architecture
- Test suite passes with or without a trained checkpoint present

## Tech Stack

| Category | Technologies |
|---|---|
| Deep Learning | PyTorch, Transfer Learning (ResNet50 / EfficientNet-B0) |
| Computer Vision | OpenCV |
| Backend | FastAPI |
| Frontend | React, Vite |
| Database | SQLite |
| Deployment | Docker |

## Project Structure

```
PlantCare-AI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/
│   │   ├── services/
│   │   └── schemas/
│   ├── train/
│   ├── data/
│   ├── tests/
│   └── Dockerfile
├── frontend/
│   ├── src/
│   └── Dockerfile
└── docker-compose.yml
```

## Model Training

ResNet50 fine-tuned on PlantVillage, evaluated with `evaluate.py` on a held-out split. 98% accuracy — full breakdown in [Model Performance](#model-performance).

Checkpoint (`best_model.pt`) isn't committed (`backend/train/checkpoints/*.pt` is gitignored). Steps below reproduce it.

**1. Get the dataset**

```bash
cd backend
pip install kagglehub
python train/download_dataset.py
```

Downloads PlantVillage via `kagglehub`, walks the (inconsistently nested) result, treats any folder containing `.jpg`/`.png` files as a class, and splits into:

```
backend/train/data/
├── train/Apple___Apple_scab/...
└── val/Apple___Apple_scab/...
```

**2. Train**

```bash
cd backend
pip install -r requirements.txt

python train/train.py \
  --data-dir train/data \
  --arch resnet50 \
  --epochs 15 \
  --output-dir train/checkpoints
```

`--output-dir` must be passed explicitly — it defaults to `checkpoints/` relative to the run directory, not `train/checkpoints/` where `app/config.py` expects it.

| Flag | Default | Purpose |
|---|---|---|
| `--data-dir` | required | Folder with `train/` and `val/` |
| `--arch` | `resnet50` | `resnet50` or `efficientnet_b0` |
| `--epochs` | 15 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 3e-4 | AdamW learning rate |
| `--image-size` | 224 | Input resolution |
| `--freeze-backbone` | off | Train only the final layer |
| `--output-dir` | `checkpoints` | Where checkpoint + class names get written |
| `--limit-per-class` | none | Cap images per class, for smoke tests |

**Pipeline details:**

- ImageNet-pretrained backbone (`torchvision.models`), final layer (`fc` / `classifier[1]`) swapped for the dataset's class count — architecture logic lives in `model_factory.py`, shared with inference
- Train-only augmentation: `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation(±15°)`, `ColorJitter`; val set gets deterministic resize + normalize
- AdamW (lr 3e-4) + `ReduceLROnPlateau` (factor 0.5, patience 2)
- `CrossEntropyLoss`
- Best-val-accuracy checkpointing to `best_model.pt`; `class_names.txt` written from `ImageFolder`'s sorted class list
- Runs on GPU if available, CPU otherwise

Output:

```
train/checkpoints/
├── best_model.pt
└── class_names.txt
```

Copy both into the path `app/config.py` expects (`MODEL_PATH` / `CLASS_NAMES_PATH`, both env-overridable) and restart the API.

## Model Evaluation

```bash
pip install scikit-learn

python train/evaluate.py \
  --data-dir train/data/val \
  --checkpoint train/checkpoints/best_model.pt \
  --class-names train/checkpoints/class_names.txt \
  --arch resnet50
```

Loads the checkpoint via `model_factory.build_model()`, runs a no-grad forward pass over validation, and prints `classification_report` / `confusion_matrix` from scikit-learn.

## Model Performance

| | |
|---|---|
| Architecture | ResNet50 (ImageNet pretrained) |
| Dataset | PlantVillage — 15 classes, 20,604 train / 12,162 val images |
| Optimizer | AdamW, lr 3e-4, ReduceLROnPlateau (0.5, patience 2) |

| Metric | Score |
|---|---|
| Accuracy | 0.98 |
| Macro Precision / Recall / F1 | 0.98 / 0.98 / 0.98 |

Weakest class: Tomato Target Spot (F1 0.93, precision 0.90) — visually confused with Early Blight and Bacterial Spot due to similar lesion patterns. Both healthy-leaf classes score F1 1.00.

Full per-class table:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Pepper — Bacterial spot | 1.00 | 0.98 | 0.99 | 571 |
| Pepper — healthy | 1.00 | 1.00 | 1.00 | 871 |
| Potato — Early blight | 1.00 | 0.99 | 0.99 | 596 |
| Potato — Late blight | 1.00 | 0.98 | 0.99 | 586 |
| Potato — healthy | 0.96 | 1.00 | 0.98 | 91 |
| Tomato — Bacterial spot | 0.99 | 0.96 | 0.97 | 1242 |
| Tomato — Early blight | 0.97 | 0.92 | 0.94 | 586 |
| Tomato — Late blight | 0.99 | 0.99 | 0.99 | 1136 |
| Tomato — Leaf Mold | 0.99 | 1.00 | 0.99 | 565 |
| Tomato — Septoria leaf spot | 0.98 | 0.99 | 0.99 | 1032 |
| Tomato — Spider mites | 0.97 | 1.00 | 0.98 | 1004 |
| Tomato — Target Spot | 0.90 | 0.97 | 0.93 | 834 |
| Tomato — Yellow Leaf Curl Virus | 1.00 | 0.99 | 1.00 | 1904 |
| Tomato — Mosaic virus | 0.99 | 0.96 | 0.97 | 220 |
| Tomato — healthy | 1.00 | 1.00 | 1.00 | 924 |

## Running Locally

**Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

`http://localhost:8000` · docs at `/docs`

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

`http://localhost:5173`

**Docker**

```bash
docker compose up --build
```

Frontend `:3000`, backend `:8000`

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/ -v
```

Smoke tests — health check, upload validation, prediction endpoint (200 with a model / 503 without), history, disease-info fallback. Pass with or without a checkpoint present.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Predict plant disease |
| GET | `/api/history` | List predictions |
| GET | `/api/history/{id}` | View a prediction |
| DELETE | `/api/history/{id}` | Delete a prediction |
| GET | `/api/health` | Health check |

## Design Decisions

- **Shared model factory** — `model_factory.py` used by training and inference, prevents checkpoint/architecture mismatch
- **Image pipeline** — OpenCV for decode + denoise (bilateral filter) + resize; torchvision for normalization, matching the training transform exactly
- **Disease knowledge base** — `disease_info.json`, keyed by class label; unknown classes fall back to a generic response. Keys must match `class_names.txt` exactly (including underscore count) — a mismatch silently falls back to the generic message instead of erroring

## Troubleshooting

**"No trained model found"** — checkpoint isn't at `train/checkpoints/best_model.pt`. Check `--output-dir` was passed during training; move the files if not.

**Treatment always shows "No entry yet"** — class name mismatch between prediction output and `disease_info.json` keys. Check with `findstr` / `grep`, fix the key, restart (disease info is `@lru_cache`d at startup).

**Local changes not showing up** — GitHub web edits only touch the remote; `git pull` first.

**`pip install` fails building pillow/pydantic-core** — `pip install --upgrade pillow pydantic pydantic-core` first, then `pip install -r requirements.txt --no-deps`.

**`torch.cuda.is_available()` is False on an NVIDIA GPU** — default PyPI torch is CPU-only:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
(`cu118` for CUDA 11.x)

## Author

Kumari Sonal — B.Sc. Computer Science and Data Analytics, IIT Patna

[GitHub](https://github.com/KS2179) · [LinkedIn](https://linkedin.com/in/kumari-sonal-a13b49329)

## License

MIT
