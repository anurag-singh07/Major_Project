# Radiosight — Kaggle Training Guide

## 📁 File: `radiosight_train.py`

---

## Step 1: Kaggle Notebook Setup

1. Go to [kaggle.com](https://kaggle.com) → **New Notebook**
2. Click **Settings** (right panel):
   - **Accelerator** → Select **GPU T4 x2** or **GPU P100**
   - **Internet** → Turn **ON** (required for kagglehub)
3. Click **File** → **Upload** → upload `radiosight_train.py`

---

## Step 2: Install Dependencies

Create a **new cell at the top** and run:

```python
!pip install -q kagglehub opencv-python-headless
```

---

## Step 3: Run the Script

In the next cell, run:

```python
exec(open('/kaggle/working/radiosight_train.py').read())
```

OR paste the entire script content directly into cells.

---

## Step 4: Monitor Training

Training will log every epoch like:
```
Epoch [01/50] | Train Loss: 0.3421 | Val Loss: 0.3102 | Val AUC: 0.7234 | Val F1: 0.2891 | Time: 342s
  ✓ Best model saved (val_loss: 0.3102)
```

- **Expected training time**: 4–8 hours on T4 GPU (full NIH dataset)
- **Checkpoints auto-save** every epoch → crash hoga to resume ho jayega

### If OOM (Out of Memory) Error:
Change in `Config` class:
```python
IMG_SIZE   = 384    # instead of 512
BATCH_SIZE = 4      # instead of 8
```

---

## Step 5: Download Artifacts

After training, go to **Output** tab → Download:

| File | Size (approx) | Purpose |
|---|---|---|
| `model.pth` | ~80 MB | Trained model weights |
| `embeddings.pkl` | ~500 MB | 1280-d vectors for all images |
| `class_names.json` | tiny | Label list |
| `test_metrics.json` | tiny | AUC, F1 scores |
| `training_curves.png` | small | Loss/AUC graphs |
| `gradcam_sample_*.png` | small | Visual Grad-CAM check |

---

## Step 6: Place Files for Backend

After downloading, put files here:

```
e:\Radioside\
├── backend\
│   ├── models\
│   │   └── model.pth          ← here
│   └── data\
│       ├── embeddings.pkl     ← here
│       └── class_names.json   ← here
```

> ✅ After this, the FastAPI backend code will be ready to use these files.

---

## Quick Debug Mode (Test Without Full Training)

To test the notebook runs without errors, set in Config:
```python
MAX_SAMPLES = 1000   # Uses only 1000 images
NUM_EPOCHS  = 2      # Only 2 epochs
```
This runs in ~10-15 minutes. Switch back to `None` and `50` for real training.
