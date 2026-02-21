# Radiosight — FastAPI Backend

## Folder Structure
```
backend/
├── main.py                     ← FastAPI app entry point
├── requirements.txt
├── .env.example                ← Copy to .env and fill values
├── models/
│   └── model.pth               ← [PUT HERE after Kaggle training]
├── data/
│   ├── embeddings.pkl          ← [PUT HERE after Kaggle training]
│   └── class_names.json        ← [PUT HERE after Kaggle training]
├── firebase_credentials.json   ← [GET from Firebase Console]
├── routes/
│   ├── predict.py              ← POST /predict
│   ├── similarity.py           ← POST /similarity
│   ├── severity.py             ← POST /severity
│   └── progression.py          ← POST /progression, GET /progression/{id}
└── utils/
    ├── model_loader.py
    ├── embedding_store.py
    ├── gradcam.py
    ├── preprocess.py
    └── firebase_utils.py
```

## Setup

### 1. Install dependencies
```bash
cd e:\Radioside\backend
pip install -r requirements.txt
```

### 2. Create .env file
```bash
copy .env.example .env
# Then edit .env with your MongoDB URI, Firebase bucket, etc.
```

### 3. Place model files (after Kaggle training)
```
backend/models/model.pth
backend/data/embeddings.pkl
backend/data/class_names.json
```

### 4. Run server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open API docs
```
http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model + embedding status |
| POST | `/predict` | Upload X-ray → prediction + heatmap + severity |
| POST | `/similarity` | 1280-d embedding → top-5 similar cases |
| POST | `/severity` | Heatmap → severity score (0-100) |
| POST | `/progression/add` | Save scan to MongoDB |
| GET | `/progression/{patient_id}` | Get patient history + trend |

## Running WITHOUT MongoDB / Firebase

The backend works without MongoDB and Firebase:
- `/predict`, `/similarity`, `/severity` work fully offline
- `/progression/*` returns 503 if MongoDB not configured
- Firebase heatmap URLs will be empty; use `heatmap_base64` instead

## Notes
- Model needs to be loaded before any `/predict` calls
- Embeddings need ~500MB RAM when loaded
- First request may be slightly slow (GradCAM hooks setup)
