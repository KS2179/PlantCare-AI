# 🌿 Plant Disease Classification

An AI-powered web app that identifies plant leaf diseases from photos.
Upload a leaf image, get a disease prediction with confidence, and see
symptoms / causes / prevention / treatment info. Every prediction is saved
to a history you can browse later.

**Stack:** PyTorch (ResNet50 / EfficientNet-B0 transfer learning) · OpenCV
(preprocessing) · FastAPI (REST API) · React + Vite (frontend) · SQLite
(prediction history) · Docker (containerized deployment)

## Why this exists

Built as a learning project to go end-to-end on a computer vision problem:
train a real model, wrap it in a production-style API, build a usable
frontend, and containerize the whole thing. The architecture is original;
the dataset (PlantVillage) is the standard public benchmark for this task.

## Project structure

```
plant-disease-classification/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app + CORS + startup
│   │   ├── config.py               # paths, model arch, DB URL
│   │   ├── models/db.py            # SQLAlchemy models (prediction history)
│   │   ├── routers/
│   │   │   ├── predict.py          # POST /api/predict
│   │   │   └── history.py          # GET/DELETE /api/history
│   │   ├── services/
│   │   │   ├── model_factory.py    # builds ResNet50 / EfficientNet-B0
│   │   │   ├── preprocessing.py    # OpenCV denoise + resize
│   │   │   ├── inference.py        # loads model, runs predictions
│   │   │   └── disease_info.py     # symptoms/causes/prevention/treatment lookup
│   │   └── schemas/prediction.py   # Pydantic response models
│   ├── train/
│   │   ├── train.py                # transfer-learning training script
│   │   └── evaluate.py             # precision/recall/F1 on a held-out set
│   ├── data/disease_info.json      # disease knowledge base (22 classes seeded)
│   ├── tests/test_api.py           # pytest smoke tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js                  # axios client
│   │   └── components/
│   │       ├── UploadForm.jsx
│   │       ├── PredictionResult.jsx
│   │       └── HistoryList.jsx
│   ├── Dockerfile
│   └── nginx.conf
└── docker-compose.yml
```

## 1. Get the dataset and train a model

This repo does **not** ship a trained model or the PlantVillage dataset
(both are large binaries). You need to train one yourself:

1. Download the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
   (or any mirror — search "PlantVillage dataset download").
2. Arrange it into this folder structure (an 80/20 train/val split):
   ```
   backend/train/data/
     train/
       Apple___Apple_scab/*.jpg
       Tomato___Late_blight/*.jpg
       ...
     val/
       Apple___Apple_scab/*.jpg
       ...
   ```
   `torchvision.datasets.ImageFolder` picks class names straight from the
   folder names, so keep them exactly as PlantVillage names them (or update
   `backend/data/disease_info.json` to match whatever names you use).
3. Train:
   ```bash
   cd backend
   pip install -r requirements.txt
   python train/train.py --data-dir train/data --arch resnet50 --epochs 15
   ```
   This writes `train/checkpoints/best_model.pt` and
   `train/checkpoints/class_names.txt` — both required by the API.
4. (Optional) Evaluate on the val set with a full classification report.
   This script needs `scikit-learn`, which isn't in the default install
   (it can fail to build on some Windows/Python combos without a C
   compiler) — install it separately if you want to run this:
   ```bash
   pip install scikit-learn
   python train/evaluate.py --data-dir train/data/val \
     --checkpoint train/checkpoints/best_model.pt \
     --class-names train/checkpoints/class_names.txt --arch resnet50
   ```

**No GPU?** Add `--freeze-backbone` to only fine-tune the final layer —
much faster on CPU, though accuracy will be lower than full fine-tuning.

## 2. Run locally (without Docker)

Backend:
```bash
cd backend
python -m venv venv
```
Activate the virtual environment (pick your shell):
```bash
# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows cmd.exe
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```
Then install and run:
```bash
pip install -r requirements-dev.txt   # includes pytest, httpx, scikit-learn on top of the runtime deps
uvicorn app.main:app --reload
```
API docs at `http://localhost:8000/docs`.

> **PowerShell note:** if `Activate.ps1` fails with a "running scripts is
> disabled" error, run PowerShell as Administrator once and execute
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try activating
> again.

Frontend:
```bash
cd frontend
npm install
npm run dev
```
App at `http://localhost:5173`.

## 3. Run with Docker

```bash
docker compose up --build
```
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

The checkpoint folder and SQLite data folder are mounted as volumes, so
train the model on your host machine first (step 1), then bring up
Docker — the API will pick up the checkpoint on container start.

## 4. Run tests

```bash
cd backend
# activate your venv first (see above) if it isn't already
pip install -r requirements-dev.txt   # skip if already installed
pytest tests/ -v
```
Tests cover routing, validation, and the disease-info lookup without
requiring a trained model; the `/api/predict` test accepts either a 200
(model present) or 503 (model not trained yet) as correct.

## API overview

| Method | Route | Description |
|---|---|---|
| POST | `/api/predict` | Upload an image, get prediction + disease info |
| GET | `/api/history?limit=&offset=` | List past predictions |
| GET | `/api/history/{id}` | Full detail for one past prediction |
| DELETE | `/api/history/{id}` | Remove a history entry |
| GET | `/api/health` | Health check |

## Notes on design decisions

- **Why a separate `model_factory.py`?** Training and inference both
  import it, so the architecture can never drift between the two —
  the classic bug where a model trains fine but inference loads the
  wrong head shape.
- **Why OpenCV *and* torchvision transforms?** OpenCV handles raw-upload
  decoding and denoising (`preprocessing.py`); torchvision handles the
  tensor normalization the model was trained with. Keeping them separate
  makes it easy to explain "where does OpenCV actually do something" —
  a fair question, since a lot of tutorial projects add it decoratively.
- **Why does `disease_info.py` never 404 on an unknown class?** If you
  train on the full 38-class PlantVillage set but the JSON knowledge base
  only covers a subset, predictions still work — you just get a "no entry
  yet" note instead of a broken response.

## Extending it

- Add the remaining PlantVillage classes to `backend/data/disease_info.json`.
- Swap `--arch efficientnet_b0` in training for a smaller/faster model.
- Add user accounts so history is per-user instead of global.
- Add a confidence threshold that flags low-confidence predictions as
  "uncertain — try a clearer photo" instead of committing to a guess.
