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

This repository does **not** include the trained model or PlantVillage dataset due to their large size.

### Dataset

Download the **PlantVillage** dataset and arrange it as follows:

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

### Train the model

```bash
cd backend

pip install -r requirements.txt

python train/train.py \
  --data-dir train/data \
  --arch resnet50 \
  --epochs 15
```

The training process generates:

```text
train/checkpoints/
├── best_model.pt
└── class_names.txt
```

---

##  Model Evaluation

(Optional)

```bash
pip install scikit-learn

python train/evaluate.py \
  --data-dir train/data/val \
  --checkpoint train/checkpoints/best_model.pt \
  --class-names train/checkpoints/class_names.txt \
  --arch resnet50
```

The evaluation script reports:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report

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

A common `model_factory.py` is used by both training and inference to ensure model architecture consistency and prevent checkpoint incompatibility.

### Image Processing Pipeline

OpenCV performs image decoding and preprocessing, while torchvision handles normalization and tensor transformations matching the model's training pipeline.

### Disease Knowledge Base

Disease descriptions are maintained separately in `disease_info.json`, allowing predictions even if metadata for a particular class has not yet been added.

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

# 🙏 Acknowledgements
This project was built using several excellent open-source tools and resources.
