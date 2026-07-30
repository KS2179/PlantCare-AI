#  PlantCare AI

An AI-powered web application for **plant disease classification** using deep learning and computer vision. Users can upload a leaf image to receive a disease prediction with a confidence score, along with detailed information about the disease including symptoms, causes, treatment, and prevention. Every prediction is stored in a SQLite database for future reference.

---

##  Features

- Plant disease classification using Deep Learning
-  Upload plant leaf images for prediction
-  Confidence score for every prediction
-  Disease information:
  - Description
  - Symptoms
  - Causes
  - Treatment
  - Prevention
-  Prediction history stored using SQLite
-  REST API built with FastAPI
-  Modern React + Vite frontend
- Docker support for containerized deployment

---

##  Tech Stack

| Category | Technologies |
|----------|--------------|
| Deep Learning | PyTorch, Transfer Learning (ResNet50 / EfficientNet-B0) |
| Computer Vision | OpenCV |
| Backend | FastAPI |
| Frontend | React, Vite |
| Database | SQLite |
| Deployment | Docker |
| Language | Python, JavaScript |

---

##  Motivation

This project was developed to gain hands-on experience in building a complete computer vision application—from model training to deployment. It demonstrates:

- Transfer learning with CNNs
- Image preprocessing using OpenCV
- REST API development with FastAPI
- Frontend integration using React
- Database management with SQLite
- Docker-based deployment

The model is trained using the publicly available **PlantVillage** dataset.

---

#  Project Structure

```text
PlantCare-AI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/
│   │   ├── routers/
│   │   ├── services/
│   │   └── schemas/
│   ├── train/
│   ├── data/
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
│
└── docker-compose.yml
```

---

# Model Training

A trained checkpoint is included in this repository at `backend/train/checkpoints/best_model.pt` (ResNet50, 98% validation accuracy across 15 disease classes — see [Model Performance](#-model-performance) below), so you can run the app immediately without training anything yourself. The training dataset itself is **not** included due to its size — see below if you want to retrain from scratch or fine-tune further.

**Quick start:** since the checkpoint is already committed, you can skip straight to [Running Locally](#-running-locally) or [Docker](#-docker) — the API will pick it up automatically on startup. The rest of this section only applies if you want to retrain the model yourself (e.g. on a different dataset, or to try a different architecture).

### 1. Get the dataset

`train/download_dataset.py` automates the download using [`kagglehub`](https://pypi.org/project/kagglehub/) and reorganizes it into the folder layout the training script expects:

```bash
cd backend
pip install kagglehub
python train/download_dataset.py
```

The first run will prompt you to authenticate with a Kaggle account (paste an API token, or log in via the browser). The PlantVillage mirror on Kaggle ships with an inconsistent nested folder structure, so the script walks the download recursively, finds every folder that directly contains `.jpg`/`.JPG`/`.png` files (treating each as one class), and performs a train/val split. The result is written to:

```text
backend/train/data/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Tomato___Late_blight/
│   └── ...
└── val/
    ├── Apple___Apple_scab/
    └── ...
```

You can also skip the script and arrange your own leaf-image dataset in this same `Plant___Disease` folder-per-class layout — `train.py` doesn't care where the images came from, only that they're organized this way.

### 2. Train the model

```bash
cd backend
pip install -r requirements.txt

python train/train.py \
  --data-dir train/data \
  --arch resnet50 \
  --epochs 15 \
  --output-dir train/checkpoints
```

> ⚠️ **Always pass `--output-dir train/checkpoints` explicitly.** The flag defaults to a plain `checkpoints` folder *relative to wherever you run the command from* — if you run `train.py` from `backend/` (as shown above), the default landing spot is `backend/checkpoints/`, **not** `backend/train/checkpoints/` where `app/config.py` looks for the model by default. Passing the flag explicitly avoids that mismatch entirely.

**What actually happens during training:**

- **Backbone & transfer learning** — the model starts from ImageNet-pretrained weights (`torchvision.models`) for either `resnet50` or `efficientnet_b0`, with only the final classification layer (`fc` for ResNet, `classifier[1]` for EfficientNet) replaced to output the number of classes found in your dataset. Architecture selection and layer-swapping live in `model_factory.py`, which both `train.py` and the inference service import — this guarantees the exact same network structure is used for training and serving, so a checkpoint trained with one config can never silently mismatch the model built for inference.
- **Data augmentation (training set only)** — `RandomResizedCrop` (scale 0.8–1.0), `RandomHorizontalFlip`, `RandomRotation(±15°)`, and `ColorJitter` (brightness/contrast/saturation at 0.2), followed by ImageNet mean/std normalization. The validation set only gets a deterministic resize + normalize, so validation accuracy reflects real generalization rather than augmented inputs.
- **Optimizer & scheduling** — `AdamW` with a default learning rate of `3e-4`, paired with `ReduceLROnPlateau` (mode `max`, factor `0.5`, patience `2` epochs) that halves the learning rate once validation accuracy stalls for 2 epochs.
- **Loss** — standard multi-class `CrossEntropyLoss` over the softmax logits.
- **Checkpointing** — after every epoch, if validation accuracy improves on the best seen so far, the full model `state_dict` is saved to `checkpoints/best_model.pt`. Class names are written once at the start of training, straight from `ImageFolder`'s alphabetically-sorted class list, to `checkpoints/class_names.txt` — the same order the model's output logits use, so inference can map logits back to labels correctly.
- **Optional backbone freezing** — pass `--freeze-backbone` to train only the final classification layer (all other parameters get `requires_grad=False`), which trains much faster and is useful when your dataset is small or you just want a quick baseline before committing to a full fine-tune.
- **Fast smoke-test runs** — `--limit-per-class N` caps each class to at most `N` training images (and a smaller automatic cap for validation) so you can verify the entire pipeline runs end-to-end in a couple of minutes before kicking off a full multi-hour training run on the complete dataset.

Full list of configurable flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--data-dir` | *required* | Folder containing `train/` and `val/` subfolders |
| `--arch` | `resnet50` | `resnet50` or `efficientnet_b0` |
| `--epochs` | `15` | Number of training epochs |
| `--batch-size` | `32` | Batch size for both train and val loaders |
| `--lr` | `3e-4` | Initial learning rate for AdamW |
| `--image-size` | `224` | Input resolution (square) |
| `--freeze-backbone` | off | Train only the final classifier layer |
| `--output-dir` | `checkpoints` | Where `best_model.pt` / `class_names.txt` are written |
| `--num-workers` | `0` | DataLoader worker processes |
| `--limit-per-class` | none | Cap images per class for a fast smoke test |

Training runs on GPU automatically if `torch.cuda.is_available()`, falling back to CPU otherwise — no config needed either way.

The training process generates:

```text
train/checkpoints/
├── best_model.pt
└── class_names.txt
```

Copy (or symlink) these two files into the path the API expects (`MODEL_PATH` / `CLASS_NAMES_PATH` in `app/config.py`, both overridable via environment variables) and restart the API to serve the newly trained model.

---

##  Model Evaluation

Once you have a checkpoint, `evaluate.py` runs it against a held-out validation folder and reports per-class metrics — useful both for sanity-checking training and for answering "how did you validate it" in a review or interview:

```bash
pip install scikit-learn

python train/evaluate.py \
  --data-dir train/data/val \
  --checkpoint train/checkpoints/best_model.pt \
  --class-names train/checkpoints/class_names.txt \
  --arch resnet50
```

Under the hood this loads the checkpoint through the same `model_factory.build_model()` used for training and inference, runs a full forward pass over the validation set with no gradient tracking, and uses scikit-learn's `classification_report` and `confusion_matrix` to print:

- Per-class **precision**, **recall**, and **F1-score**
- Overall **accuracy**
- Confusion matrix shape (`num_classes × num_classes`), so you can spot which diseases are most often confused with each other

Because this project doesn't ship a pretrained checkpoint or bundled dataset, exact accuracy numbers will depend on the dataset split, architecture, and epoch count you train with — run the evaluation yourself after training to get numbers specific to your run rather than relying on a generic figure.

---

##  Model Performance

**Training setup:**

| | |
|---|---|
| Architecture | ResNet50 (ImageNet pretrained) |
| Dataset | PlantVillage — 15 classes (Pepper, Potato, Tomato), 20,604 training images, 12,162 validation images |
| Optimizer | AdamW, lr `3e-4`, with `ReduceLROnPlateau` (factor 0.5, patience 2) |
| Augmentation | Random resized crop, horizontal flip, rotation (±15°), color jitter |
| Hardware | Local NVIDIA GPU |

**Overall metrics** (from `evaluate.py` on the held-out validation set, 12,162 images):

| Metric | Score |
|--------|-------|
| Accuracy | **0.98** |
| Macro Precision | 0.98 |
| Macro Recall | 0.98 |
| Macro F1-score | 0.98 |
| Weighted Precision/Recall/F1 | 0.98 / 0.98 / 0.98 |

**Per-class breakdown:**

| Class | Precision | Recall | F1-score | Support |
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
| Tomato — Spider mites (two-spotted) | 0.97 | 1.00 | 0.98 | 1004 |
| Tomato — Target Spot | 0.90 | 0.97 | 0.93 | 834 |
| Tomato — Yellow Leaf Curl Virus | 1.00 | 0.99 | 1.00 | 1904 |
| Tomato — Mosaic virus | 0.99 | 0.96 | 0.97 | 220 |
| Tomato — healthy | 1.00 | 1.00 | 1.00 | 924 |

The model reaches **98% overall accuracy** on the held-out validation set. Both healthy-leaf classes (Pepper and Tomato) are classified essentially perfectly (F1 = 1.00), which is expected since healthy foliage tends to look visually distinct from any of the disease classes.

The hardest class is **Tomato Target Spot** (F1 = 0.93) — its precision (0.90) is the lowest of any class, meaning roughly 10% of images the model labeled "Target Spot" actually belonged to a different class. This tracks with the domain: target spot produces small necrotic lesions that visually resemble early blight and bacterial spot on tomato leaves, so some confusion between these visually similar spot/lesion diseases is expected. Tomato Early Blight (F1 = 0.94) and Tomato Mosaic Virus (F1 = 0.97) are the next hardest, plausibly for the same reason — Early Blight shares the lesion-based symptom pattern with Target Spot.

**Confusion matrix:** `evaluate.py` currently prints only the matrix's shape (`15 × 15`) rather than its values. To get the actual class-vs-class breakdown — useful for confirming exactly which classes Target Spot gets confused with — add a couple of lines to save it as an image:

```python
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("docs/confusion_matrix.png")
```

Then embed it here:
```markdown
![Confusion matrix](docs/confusion_matrix.png)
```

**Sample predictions:** these are real results from the running app (screenshots taken during testing):

| Input | Predicted | Confidence |
|-------|-----------|------------|
| Tomato leaf with dark circular spots (Septoria leaf spot) | Tomato Septoria Leaf Spot | 98.1% |
| Wilted tomato foliage with brown lesions | Tomato Early Blight | 92.1% / 89.1% (two separate photos) |

> To make this section complete, save the actual test images into a `docs/samples/` folder in the repo and reference them here with `![description](docs/samples/filename.jpg)` — that turns this from a text table into an actual visual proof the model works, which is worth more to a reviewer than the numbers alone.

---

#  Running Locally

## Backend

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements-dev.txt

uvicorn app.main:app --reload
```

Backend:

```
http://localhost:8000
```

Swagger Documentation:

```
http://localhost:8000/docs
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend:

```
http://localhost:5173
```

---

#  Docker

Run the complete application using Docker:

```bash
docker compose up --build
```

After startup:

Frontend

```
http://localhost:3000
```

Backend

```
http://localhost:8000
```

---

#  Running Tests

```bash
cd backend

pip install -r requirements-dev.txt

pytest tests/ -v
```

The test suite (`tests/test_api.py`) is written as smoke tests that pass even without a trained checkpoint present — it covers the health check, upload validation (rejecting non-image content types), the prediction endpoint (expecting either a `200` with a real model or a clear `503` without one), history listing, and disease-info lookups (including graceful fallback for undocumented classes).

---

# 📡 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/predict` | Predict plant disease |
| GET | `/api/history` | Retrieve prediction history |
| GET | `/api/history/{id}` | View a prediction |
| DELETE | `/api/history/{id}` | Delete a prediction |
| GET | `/api/health` | Health check |

---

#  Design Decisions

### Shared Model Factory

A common `model_factory.py` is used by both training and inference to ensure model architecture consistency and prevent checkpoint incompatibility — the single most common cause of "works during training, garbage at inference" bugs.

### Image Processing Pipeline

OpenCV performs image decoding and light preprocessing (a bilateral filter for denoising phone-camera photos, then resizing), while torchvision handles normalization and tensor transformations matching the model's training pipeline exactly.

### Disease Knowledge Base

Disease descriptions (symptoms, causes, prevention, treatment) are maintained separately in `disease_info.json`, keyed by class label, so the API can serve predictions even for classes that haven't been documented yet — it falls back to a generic response derived from the class name rather than failing.

> **Note:** the lookup in `disease_info.py` is an exact string match against the model's class names (no normalization), so every key in `disease_info.json` must match the corresponding entry in `train/checkpoints/class_names.txt` **character for character** — including underscore count. If you retrain on a different dataset variant with different folder-naming conventions, double check the keys still line up, or predictions will silently fall back to the generic "no entry yet" message even though the disease was correctly identified.

---

#  Troubleshooting

**"No trained model found at ..." on API startup**
`app/config.py` looks for the checkpoint at `train/checkpoints/best_model.pt` by default. If you trained without passing `--output-dir train/checkpoints`, your checkpoint likely landed in a different folder (see the warning in the Model Training section above). Move `best_model.pt` and `class_names.txt` into `backend/train/checkpoints/` and restart the API.

**Predictions work, but "Treatment" always says "No entry yet in the disease knowledge base"**
This means the predicted class label isn't found as an exact key in `data/disease_info.json` — usually because the naming convention doesn't match `class_names.txt` (e.g. `Tomato_Early_blight` vs. `Tomato___Early_blight`). Check with:
```bash
type data\disease_info.json | findstr "YourClassNameHere"
```
If nothing matches, fix the key in `disease_info.json` to match `class_names.txt` exactly, then **restart the backend** — the file is loaded once at startup and cached (`@lru_cache`), so saving the file alone isn't enough.

**Editing files on GitHub but changes don't show up locally**
Editing a file through the GitHub website only updates the remote repository, not your local clone. Run `git pull` in your project folder to bring those changes down before restarting the backend.

**`pip install -r requirements.txt` fails trying to build `pillow`/`pydantic-core` from source**
This usually means pip is falling back to compiling an old pinned version because no prebuilt wheel matches your Python version. Install newer versions of just those two packages first, then install the rest without letting pip downgrade them:
```bash
pip install --upgrade pillow pydantic pydantic-core
pip install -r requirements.txt --no-deps
```

**`torch.cuda.is_available()` returns `False` despite having an NVIDIA GPU**
The default PyPI build of `torch` is CPU-only. Check your CUDA version with `nvidia-smi`, then reinstall from PyTorch's own index:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
(use `cu118` instead of `cu124` if `nvidia-smi` reports CUDA 11.x)

---


#  Author

**Kumari Sonal**

B.Sc. Computer Science and Data Analytics  
Indian Institute of Technology Patna

GitHub: https://github.com/KS2179

LinkedIn: https://linkedin.com/in/kumari-sonal-a13b49329

---

# 📄 License

This project is licensed under the **MIT License**.

---

#  Acknowledgements
This project was built using several excellent open-source tools and resources.
