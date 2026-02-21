"""
╔══════════════════════════════════════════════════════════════════╗
║         RADIOSIGHT — Explainable Chest X-ray Training            ║
║         Dataset   : NIH ChestX-ray14 (via kagglehub)            ║
║         Model     : EfficientNet-V2-S (Multi-label)              ║
║         Platform  : Kaggle (T4 GPU)                              ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO USE THIS ON KAGGLE:
1. Create a new Kaggle Notebook
2. Enable GPU: Settings → Accelerator → GPU (T4 x2 or P100)
3. Enable Internet: Settings → Internet → On   (required for kagglehub)
4. Upload this file or paste code into cells
5. Run all cells — training will auto-resume if interrupted
6. After training: Download model.pth, embeddings.pkl, class_names.json
   from /kaggle/working/ (Output section)
"""

# ─────────────────────────────────────────────────
# SECTION 0 — Install required packages
# ─────────────────────────────────────────────────
# Run this first in Kaggle (in a separate cell):
# !pip install -q kagglehub timm

# ─────────────────────────────────────────────────
# SECTION 1 — Imports
# ─────────────────────────────────────────────────
import os
import sys
import gc
import json
import time
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
# PyTorch 2.x compatible imports
try:
    from torch.amp import GradScaler, autocast  # PyTorch 2.x
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch 1.x fallback

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (Kaggle-safe)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Prevent PIL from crashing on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
# SECTION 2 — Config (all hyperparams in one place)
# ─────────────────────────────────────────────────
class Config:
    # ------- Paths -------
    DATASET_SLUG   = "nih-chest-xrays/data"         # kagglehub slug
    WORK_DIR       = "/kaggle/working"               # output dir
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"   # auto-save dir

    # ------- Image -------
    IMG_SIZE       = 512          # 512×512 (change to 384 if OOM)
    CHANNELS       = 3

    # ------- Dataset -------
    TRAIN_FRAC     = 0.70
    VAL_FRAC       = 0.15
    # TEST = remaining 0.15 (automatically)
    MAX_SAMPLES    = None         # Set e.g. 10000 for quick debug; None = full dataset

    # ------- Training -------
    BATCH_SIZE     = 8            # 8 is safe for T4+512px; lower if OOM
    NUM_EPOCHS     = 50
    LR             = 1e-4
    WEIGHT_DECAY   = 1e-5
    GRAD_CLIP      = 1.0          # prevents exploding gradients

    # ------- Early Stopping -------
    PATIENCE       = 7            # epochs to wait before stopping
    MIN_DELTA      = 1e-4         # minimum improvement to count

    # ------- Model -------
    MODEL_NAME     = "efficientnet_v2_s"
    EMBEDDING_DIM  = 1280         # penultimate layer output for EfficientNet-V2-S
    DROPOUT        = 0.4

    # ------- Workers -------
    NUM_WORKERS    = 2            # keep low on Kaggle to avoid crashes

    # ------- Labels (NIH ChestX-ray14) -------
    DISEASE_LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    NUM_CLASSES    = len(DISEASE_LABELS)  # 14

    # ------- Misc -------
    SEED           = 42
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP        = True         # mixed precision — saves GPU memory

cfg = Config()

# ─────────────────────────────────────────────────
# SECTION 3 — Logging & Seeding
# ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{cfg.WORK_DIR}/training.log")
    ]
)
log = logging.getLogger("Radiosight")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.SEED)
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

log.info(f"Using device: {cfg.DEVICE}")
log.info(f"AMP (Mixed Precision): {cfg.USE_AMP}")
if cfg.DEVICE == "cuda":
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────────
# SECTION 4 — Dataset Download (kagglehub)
# ─────────────────────────────────────────────────
log.info("=" * 60)
log.info("Downloading NIH ChestX-ray14 dataset via kagglehub...")
log.info("=" * 60)

try:
    import kagglehub
    dataset_root = kagglehub.dataset_download("nih-chest-xrays/data")
    log.info(f"Dataset path: {dataset_root}")
except Exception as e:
    log.error(f"kagglehub download failed: {e}")
    log.error("Make sure: 1) Internet is ON in Kaggle settings 2) kagglehub is installed")
    raise

# Verify CSV exists
csv_path = os.path.join(dataset_root, "Data_Entry_2017.csv")
if not os.path.exists(csv_path):
    # Try alternate structure
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            if "Data_Entry" in f:
                csv_path = os.path.join(root, f)
                break

log.info(f"CSV found: {csv_path}")

# ─────────────────────────────────────────────────
# SECTION 5 — CSV Parsing & Label Engineering
# ─────────────────────────────────────────────────
log.info("Parsing CSV and building label matrix...")

df = pd.read_csv(csv_path)
log.info(f"Total entries in CSV: {len(df)}")
log.info(f"Columns: {list(df.columns)}")

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# Find image index column
img_col = None
for col in ["Image Index", "image_index", "filename", "Image_Index"]:
    if col in df.columns:
        img_col = col
        break
if img_col is None:
    img_col = df.columns[0]
    log.warning(f"Using first column as image index: {img_col}")

# Find label column
label_col = None
for col in ["Finding Labels", "finding_labels", "labels"]:
    if col in df.columns:
        label_col = col
        break
if label_col is None:
    log.error("Could not find 'Finding Labels' column!")
    log.error(f"Available columns: {list(df.columns)}")
    raise ValueError("Label column not found")

log.info(f"Image col: '{img_col}' | Label col: '{label_col}'")

# Build multi-hot label matrix
def encode_labels(label_str: str, all_labels: list) -> np.ndarray:
    """Convert 'Atelectasis|Effusion' to [1,0,0,1,...,0]"""
    parts = [p.strip() for p in str(label_str).split("|")]
    vec = np.zeros(len(all_labels), dtype=np.float32)
    for i, lbl in enumerate(all_labels):
        if lbl in parts:
            vec[i] = 1.0
    return vec

# Apply encoding
label_matrix = np.stack([
    encode_labels(row[label_col], cfg.DISEASE_LABELS)
    for _, row in df.iterrows()
])
df['label_vec'] = list(label_matrix)

# Mark "Normal" (No Finding) — useful for stats
df['is_normal'] = df[label_col].apply(lambda x: 'No Finding' in str(x))

log.info(f"Normal samples : {df['is_normal'].sum():,}")
log.info(f"Disease samples: {(~df['is_normal']).sum():,}")

# Save class names for later use
class_names = cfg.DISEASE_LABELS
with open(f"{cfg.WORK_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
log.info(f"class_names.json saved → {cfg.WORK_DIR}/class_names.json")

# ─────────────────────────────────────────────────
# SECTION 6 — Build Image Path Map
# ─────────────────────────────────────────────────
log.info("Building image path index... (may take a minute)")

# NIH dataset structure: images/images_001/images/*.png
image_paths = {}
for fpath in glob(os.path.join(dataset_root, "**", "*.png"), recursive=True):
    fname = os.path.basename(fpath)
    image_paths[fname] = fpath

log.info(f"Found {len(image_paths):,} PNG images on disk")

# Filter df to only rows with images on disk
df['filepath'] = df[img_col].map(image_paths)
df_valid = df[df['filepath'].notna()].reset_index(drop=True)
missing = len(df) - len(df_valid)
log.info(f"Valid (found on disk): {len(df_valid):,} | Missing: {missing:,}")

if cfg.MAX_SAMPLES:
    df_valid = df_valid.sample(min(cfg.MAX_SAMPLES, len(df_valid)),
                               random_state=cfg.SEED).reset_index(drop=True)
    log.info(f"DEBUG MODE: using {len(df_valid):,} samples")

# ─────────────────────────────────────────────────
# SECTION 7 — Train/Val/Test Split
# ─────────────────────────────────────────────────
# First split: train vs rest
train_df, temp_df = train_test_split(
    df_valid, test_size=(1 - cfg.TRAIN_FRAC),
    random_state=cfg.SEED, shuffle=True
)
# Second split: val vs test (equal halves of temp)
val_size_ratio = cfg.VAL_FRAC / (1 - cfg.TRAIN_FRAC)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5,
    random_state=cfg.SEED, shuffle=True
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

log.info(f"Split → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# ─────────────────────────────────────────────────
# SECTION 8 — Transforms & Augmentation
# ─────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE),
                      interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),          # ensure 3-channel
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # subtle augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE),
                      interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────────
# SECTION 9 — Dataset Class
# ─────────────────────────────────────────────────
class NIHChestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, mode: str = "train"):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.mode      = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = row['filepath']
        label    = torch.tensor(row['label_vec'], dtype=torch.float32)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"[Dataset] Could not open {img_path}: {e} — using black image")
            img = Image.new("RGB", (cfg.IMG_SIZE, cfg.IMG_SIZE), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label, str(row[img_col])  # img, label, image_id

def create_dataloaders():
    train_ds = NIHChestDataset(train_df, train_transform, mode="train")
    val_ds   = NIHChestDataset(val_df,   val_transform,   mode="val")
    test_ds  = NIHChestDataset(test_df,  val_transform,   mode="test")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0), prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0)
    )
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

# ─────────────────────────────────────────────────
# SECTION 10 — Model Architecture
# ─────────────────────────────────────────────────
class RadiosightModel(nn.Module):
    """
    EfficientNet-V2-S backbone + custom multi-label classification head.
    Penultimate layer outputs 1280-d embedding for similarity search.
    """
    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        # Load pretrained EfficientNet-V2-S
        backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        # Keep all feature layers (conv blocks)
        self.features = backbone.features          # outputs (B, 1280, H', W')
        self.avgpool  = backbone.avgpool           # Global Avg Pool → (B, 1280)

        # Custom classifier head with dropout
        self.embedding_dim = 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embedding_dim, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)                   # (B, 1280, H', W')
        pooled = self.avgpool(feat)                # (B, 1280, 1, 1)
        embedding = torch.flatten(pooled, 1)       # (B, 1280)  ← this is our embedding
        logits = self.classifier(embedding)        # (B, num_classes)
        return logits, embedding

    def get_embedding(self, x):
        """Extract embedding only (no classification head)"""
        with torch.no_grad():
            feat = self.features(x)
            pooled = self.avgpool(feat)
            embedding = torch.flatten(pooled, 1)
        return embedding

    def get_last_conv_layer(self):
        """Return last convolutional layer (for Grad-CAM)"""
        # Last block of EfficientNet-V2-S features
        return self.features[-1]


def build_model() -> RadiosightModel:
    model = RadiosightModel(num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT)
    model = model.to(cfg.DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {cfg.MODEL_NAME}")
    log.info(f"Total params    : {total_params:,}")
    log.info(f"Trainable params: {trainable_params:,}")
    return model

# ─────────────────────────────────────────────────
# SECTION 11 — Loss Function (Weighted BCE for Imbalance)
# ─────────────────────────────────────────────────
def compute_pos_weights(df: pd.DataFrame, device) -> torch.Tensor:
    """
    Compute positive class weights to handle class imbalance.
    pos_weight[i] = (N - pos_i) / pos_i  (inverse frequency)
    Clipped to [1, 50] to prevent extreme values.
    """
    label_matrix = np.stack(df['label_vec'].values)
    N = len(label_matrix)
    pos_counts = label_matrix.sum(axis=0).clip(min=1)
    neg_counts = N - pos_counts
    weights = (neg_counts / pos_counts).clip(1, 50)
    log.info("Positive class weights (imbalance correction):")
    for lbl, w in zip(cfg.DISEASE_LABELS, weights):
        log.info(f"  {lbl:<22}: {w:.1f}x")
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ─────────────────────────────────────────────────
# SECTION 12 — Metrics
# ─────────────────────────────────────────────────
def compute_metrics(all_labels: np.ndarray, all_probs: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """
    Compute per-class and mean metrics.
    all_labels: (N, 14) binary
    all_probs : (N, 14) sigmoid probabilities
    """
    all_preds = (all_probs >= threshold).astype(int)

    metrics = {}
    auc_scores = []
    f1_scores  = []

    for i, label in enumerate(cfg.DISEASE_LABELS):
        col_labels = all_labels[:, i]
        col_probs  = all_probs[:, i]
        col_preds  = all_preds[:, i]

        # AUC needs both classes present
        if col_labels.sum() > 0 and (1 - col_labels).sum() > 0:
            auc = roc_auc_score(col_labels, col_probs)
        else:
            auc = float('nan')

        f1 = f1_score(col_labels, col_preds, zero_division=0)
        auc_scores.append(auc)
        f1_scores.append(f1)

    # Filter NaN AUC
    valid_auc = [a for a in auc_scores if not np.isnan(a)]
    metrics['mean_auc'] = np.mean(valid_auc) if valid_auc else 0.0
    metrics['mean_f1']  = np.mean(f1_scores)
    metrics['per_class_auc'] = dict(zip(cfg.DISEASE_LABELS, auc_scores))
    metrics['per_class_f1']  = dict(zip(cfg.DISEASE_LABELS, f1_scores))
    return metrics

# ─────────────────────────────────────────────────
# SECTION 13 — Checkpointing (Crash Recovery)
# ─────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_DIR, "latest_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(cfg.WORK_DIR, "best_model.pth")

def save_checkpoint(model, optimizer, scaler, epoch, best_val_loss,
                    patience_counter, history):
    state = {
        'epoch'           : epoch,
        'model_state'     : model.state_dict(),
        'optimizer_state' : optimizer.state_dict(),
        'scaler_state'    : scaler.state_dict() if scaler else None,
        'best_val_loss'   : best_val_loss,
        'patience_counter': patience_counter,
        'history'         : history,
    }
    torch.save(state, CHECKPOINT_PATH)
    log.info(f"  [Checkpoint] Saved → epoch {epoch + 1}")

def load_checkpoint(model, optimizer, scaler):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, float('inf'), 0, {'train_loss': [], 'val_loss': [],
                                    'val_auc': [], 'val_f1': []}
    log.info(f"[Resume] Loading checkpoint from: {CHECKPOINT_PATH}")
    state = torch.load(CHECKPOINT_PATH, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    if scaler and state.get('scaler_state'):
        scaler.load_state_dict(state['scaler_state'])
    start_epoch      = state['epoch'] + 1
    best_val_loss    = state.get('best_val_loss', float('inf'))
    patience_counter = state.get('patience_counter', 0)
    history          = state.get('history', {'train_loss': [], 'val_loss': [],
                                              'val_auc': [], 'val_f1': []})
    log.info(f"[Resume] Resuming from epoch {start_epoch + 1}")
    return start_epoch, best_val_loss, patience_counter, history

# ─────────────────────────────────────────────────
# SECTION 14 — Training Loop
# ─────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    running_loss = 0.0
    num_batches  = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False,
                dynamic_ncols=True)
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(device_type='cuda', enabled=cfg.USE_AMP):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Periodic GPU cache clear (every 200 batches)
        if (batch_idx + 1) % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_loss = running_loss / num_batches
    return avg_loss

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels   = []
    all_probs    = []

    pbar = tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True)
    for images, labels, _ in pbar:
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        with autocast(device_type='cuda', enabled=cfg.USE_AMP):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        running_loss    += loss.item()
        probs            = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs)

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    avg_loss   = running_loss / len(loader)
    metrics    = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics, all_labels, all_probs

def full_training_pipeline():
    log.info("=" * 60)
    log.info("Building dataloaders...")
    log.info("=" * 60)
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders()
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    log.info("Building model...")
    model = build_model()

    pos_weights = compute_pos_weights(train_df, cfg.DEVICE)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer   = optim.Adam(model.parameters(), lr=cfg.LR,
                             weight_decay=cfg.WEIGHT_DECAY)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scaler      = GradScaler(device='cuda') if cfg.USE_AMP and cfg.DEVICE == "cuda" else None

    # -- Resume from checkpoint if exists --
    start_epoch, best_val_loss, patience_counter, history = load_checkpoint(
        model, optimizer, scaler
    )

    log.info("=" * 60)
    log.info(f"Training: {cfg.NUM_EPOCHS} epochs | Starting from epoch {start_epoch + 1}")
    log.info("=" * 60)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, scaler, epoch)
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            log.info(f"  [LR Scheduler] LR reduced: {old_lr:.2e} → {new_lr:.2e}")

        elapsed = time.time() - t0
        log.info(
            f"Epoch [{epoch+1:02d}/{cfg.NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_metrics['mean_auc']:.4f} | "
            f"Val F1: {val_metrics['mean_f1']:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['mean_auc'])
        history['val_f1'].append(val_metrics['mean_f1'])

        # Save checkpoint every epoch (crash recovery)
        save_checkpoint(model, optimizer, scaler, epoch,
                        best_val_loss, patience_counter, history)

        # Save best model
        if val_loss < best_val_loss - cfg.MIN_DELTA:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            log.info(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            log.info(f"  No improvement. Patience: {patience_counter}/{cfg.PATIENCE}")

        # Early stopping
        if patience_counter >= cfg.PATIENCE:
            log.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("=" * 60)
    log.info("Training complete!")
    log.info(f"Best model saved to: {BEST_MODEL_PATH}")
    log.info("=" * 60)

    # --- Final test evaluation ---
    log.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=cfg.DEVICE, weights_only=True))
    test_loss, test_metrics, test_labels, test_probs = validate(
        model, test_loader, criterion
    )
    log.info(f"TEST → Loss: {test_loss:.4f} | AUC: {test_metrics['mean_auc']:.4f} | F1: {test_metrics['mean_f1']:.4f}")
    log.info("\nPer-class AUC:")
    for cls, auc in test_metrics['per_class_auc'].items():
        log.info(f"  {cls:<22}: {auc:.4f}" if not np.isnan(auc) else f"  {cls:<22}: N/A")

    # Save metrics
    with open(f"{cfg.WORK_DIR}/test_metrics.json", "w") as f:
        safe_metrics = {}
        for k, v in test_metrics.items():
            if isinstance(v, dict):
                safe_metrics[k] = {kk: (float(vv) if not np.isnan(vv) else None)
                                   for kk, vv in v.items()}
            else:
                safe_metrics[k] = float(v) if not np.isnan(v) else None
        json.dump(safe_metrics, f, indent=2)
    log.info(f"Metrics saved → {cfg.WORK_DIR}/test_metrics.json")

    # Save training curves
    plot_training_curves(history)

    return model, history, test_labels, test_probs

# ─────────────────────────────────────────────────
# SECTION 15 — Training Curve Plot
# ─────────────────────────────────────────────────
def plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs, history['val_auc'], 'g-o')
    axes[1].set_title('Val ROC-AUC (Mean)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim([0, 1])

    axes[2].plot(epochs, history['val_f1'], 'm-o')
    axes[2].set_title('Val F1 (Mean)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f"{cfg.WORK_DIR}/training_curves.png", dpi=100)
    plt.close()
    log.info(f"Training curves saved → {cfg.WORK_DIR}/training_curves.png")

# ─────────────────────────────────────────────────
# SECTION 16 — Grad-CAM
# ─────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-V2-S.
    Generates heatmap from the last convolutional block.
    """
    def __init__(self, model: RadiosightModel):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook on last conv block (use full_backward_hook for PyTorch 2.x)
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t. the output of this layer
        self.gradients = grad_output[0].detach().clone()

    def generate(self, image_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        Args:
            image_tensor: (1, 3, H, W) preprocessed image on DEVICE
            class_idx   : target class index (if None, uses argmax of sigmoid)
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.to(cfg.DEVICE)

        # Forward pass
        self.model.zero_grad()
        logits, _ = self.model(image_tensor)
        probs      = torch.sigmoid(logits)

        if class_idx is None:
            class_idx = probs.argmax().item()

        # Backward pass for target class
        target = logits[0, class_idx]
        target.backward()

        # Compute Grad-CAM
        grads  = self.gradients[0]         # (C, H', W')
        acts   = self.activations[0]       # (C, H', W')
        weights = grads.mean(dim=(1, 2))   # Global avg pool of gradients

        # Weighted sum of activations
        cam = torch.zeros(acts.shape[1:], device=cfg.DEVICE)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)              # ReLU → only positive influence
        cam = cam.cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Resize to input image size
        cam = cv2.resize(cam, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        return cam

def overlay_heatmap(original_img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    Args:
        original_img: (H, W, 3) uint8
        heatmap     : (H, W) float [0, 1]
    Returns:
        overlay: (H, W, 3) uint8
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap      = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap      = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay       = np.uint8(alpha * colormap + (1 - alpha) * original_img)
    return overlay

def save_gradcam_sample(model: RadiosightModel, dataset: NIHChestDataset,
                        num_samples: int = 5):
    """Save sample Grad-CAM outputs for visual inspection."""
    grad_cam = GradCAM(model)
    saved    = 0
    indices  = np.random.choice(len(dataset), size=min(num_samples * 5, len(dataset)),
                                replace=False)

    for idx in indices:
        if saved >= num_samples:
            break
        img_tensor, label, img_id = dataset[idx]
        if label.sum() == 0:          # skip Normal for interesting visuals
            continue

        # Get Grad-CAM
        input_t  = img_tensor.unsqueeze(0).to(cfg.DEVICE)
        heatmap  = grad_cam.generate(input_t)

        # Denormalize image for display
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_disp = (img_tensor * std + mean).permute(1, 2, 0).numpy()
        img_disp = np.clip(img_disp * 255, 0, 255).astype(np.uint8)

        overlay = overlay_heatmap(img_disp, heatmap)

        # Save side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_disp, cmap='gray');   axes[0].set_title("Original X-ray");    axes[0].axis('off')
        axes[1].imshow(heatmap, cmap='jet');     axes[1].set_title("Grad-CAM Heatmap");  axes[1].axis('off')
        axes[2].imshow(overlay);                 axes[2].set_title("Overlay");            axes[2].axis('off')

        labels_active = [cfg.DISEASE_LABELS[i] for i in range(14) if label[i] > 0]
        plt.suptitle(f"ID: {img_id} | Labels: {', '.join(labels_active)}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{cfg.WORK_DIR}/gradcam_sample_{saved}.png", dpi=80)
        plt.close()
        saved += 1

    log.info(f"Grad-CAM samples saved → {cfg.WORK_DIR}/gradcam_sample_*.png")

# ─────────────────────────────────────────────────
# SECTION 17 — Severity Scoring
# ─────────────────────────────────────────────────
def compute_severity_score(heatmap: np.ndarray,
                           threshold: float = 0.5) -> float:
    """
    Severity = 100 * (0.7 * Afraction + 0.3 * Imean)

    Args:
        heatmap  : (H, W) float [0, 1]
        threshold: pixel value threshold to consider "activated"
    Returns:
        severity : float in [0, 100]
    """
    activated_mask = (heatmap >= threshold).astype(np.float32)
    Afraction      = activated_mask.mean()          # proportion of activated pixels
    Imean          = heatmap.mean()                 # mean activation intensity
    severity       = 100.0 * (0.7 * Afraction + 0.3 * Imean)
    severity       = float(np.clip(severity, 0, 100))
    return severity

# ─────────────────────────────────────────────────
# SECTION 18 — Embedding Extraction (All Images)
# ─────────────────────────────────────────────────
@torch.no_grad()
def extract_all_embeddings(model: RadiosightModel,
                           df_all: pd.DataFrame) -> list:
    """
    Extract 1280-d embeddings for all images in df_all.
    Returns list of dicts for MongoDB / pickle storage.
    Memory-efficient: processes in batches.
    """
    model.eval()
    db_records  = []
    batch_size  = cfg.BATCH_SIZE
    total       = len(df_all)
    n_batches   = (total + batch_size - 1) // batch_size

    log.info(f"Extracting embeddings for {total:,} images in {n_batches} batches...")

    # Temporary dataset with val_transform (no augmentation)
    embed_ds     = NIHChestDataset(df_all, val_transform, mode="embed")
    embed_loader = DataLoader(embed_ds, batch_size=batch_size,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    for images, labels, img_ids in tqdm(embed_loader, desc="Extracting embeddings"):
        images = images.to(cfg.DEVICE, non_blocking=True)
        with autocast(device_type='cuda', enabled=cfg.USE_AMP):
            _, embeddings = model(images)

        embeddings_np = embeddings.cpu().float().numpy()
        labels_np     = labels.numpy()

        for i in range(len(img_ids)):
            active_labels = [cfg.DISEASE_LABELS[j] for j in range(14)
                             if labels_np[i][j] > 0]
            record = {
                "image_id"  : img_ids[i],
                "embedding" : embeddings_np[i].tolist(),   # 1280-d list
                "label"     : active_labels if active_labels else ["No Finding"],
                "label_vec" : labels_np[i].tolist(),       # binary vector
                "metadata"  : {}
            }
            db_records.append(record)

    return db_records

# ─────────────────────────────────────────────────
# SECTION 19 — Export Artifacts
# ─────────────────────────────────────────────────
def export_all_artifacts(model: RadiosightModel, db_records: list):
    """
    Save all artifacts needed for the FastAPI backend:
    1. model.pth         — trained model weights
    2. embeddings.pkl    — all training image embeddings
    3. class_names.json  — label list (already saved)
    """
    # 1. Save final model (also saves best_model.pth during training)
    final_model_path = os.path.join(cfg.WORK_DIR, "model.pth")
    torch.save(model.state_dict(), final_model_path)
    log.info(f"model.pth → {final_model_path}")

    # 2. Save embeddings
    embeddings_path = os.path.join(cfg.WORK_DIR, "embeddings.pkl")
    with open(embeddings_path, "wb") as f:
        pickle.dump(db_records, f, protocol=4)
    log.info(f"embeddings.pkl → {embeddings_path} ({len(db_records):,} records)")

    # 3. class_names.json already saved in Section 5

    # 4. Summary
    log.info("=" * 60)
    log.info("EXPORT COMPLETE — Download these files from /kaggle/working/:")
    log.info(f"  ✓ model.pth        ({os.path.getsize(final_model_path)/1e6:.1f} MB)")
    log.info(f"  ✓ embeddings.pkl   ({os.path.getsize(embeddings_path)/1e6:.1f} MB)")
    log.info(f"  ✓ class_names.json")
    log.info(f"  ✓ test_metrics.json")
    log.info(f"  ✓ training_curves.png")
    log.info("=" * 60)

# ─────────────────────────────────────────────────
# SECTION 20 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        log.info("▶ Starting Radiosight training pipeline...")

        # STEP 1: Train the model
        model, history, test_labels, test_probs = full_training_pipeline()

        # STEP 2: Grad-CAM samples (visual check)
        log.info("▶ Generating Grad-CAM samples...")
        _, _, _, train_ds, val_ds, _ = create_dataloaders()
        save_gradcam_sample(model, val_ds, num_samples=5)

        # STEP 3: Extract embeddings for all training + val images
        #         (test excluded — unseen data)
        log.info("▶ Extracting embeddings from train+val set...")
        df_trainval = pd.concat([train_df, val_df]).reset_index(drop=True)
        db_records  = extract_all_embeddings(model, df_trainval)

        # STEP 4: Export all artifacts
        log.info("▶ Exporting artifacts...")
        export_all_artifacts(model, db_records)

        log.info("✅ ALL DONE! Training pipeline completed successfully.")

    except KeyboardInterrupt:
        log.warning("Training interrupted by user (Ctrl+C). Checkpoint is saved.")

    except torch.cuda.OutOfMemoryError:
        log.error("CUDA Out of Memory!")
        log.error("Fix: Reduce IMG_SIZE to 384 or BATCH_SIZE to 4 in Config class.")
        raise

    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        log.error("Check training.log for details. Last checkpoint is saved.")
        raise
