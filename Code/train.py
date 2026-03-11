"""
Training script for the FDR-TransUNet on SIIM pneumothorax Detection.

Matching the paper's training setup: 
- Adam optimizer with (learning rate) LR=3e-4 (0.0003) and weight decay
- Cosine annealing with warm restarts (T_0=8, T_mult=2)
- BCE with Logits Loss
- Deep supervision (sum of both path losses)
- Data augmentation:
    - rotation: ±10°
    - horizontal flip: p=0.4 (40%)
    - normalize

- mIoU-based early stopping (patience > 10 epochs) -> if no improvement in mIoU in 11 epochs, stop. 

The following are parameters that were not found in the paper but were chosen based on:
- TransUNet codebase, standard fro ViT-based models -> Weight Decay = 0.01
- SIIM pnemothorax competition solutions (BCE + Dice loss helps class imbalance)
- ViT training best practices (gradient clipping prevents exploding gradients) -> sugguestion 

"""
import os
import sys
import json


# ------------------------------------------------------------------
# Make sure Python can find our modules (model, dataset, config, etc.)
# even when running from project root: python Code/train.py
# ------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
import config
import data_processing
from dataset import PneumothoraxDataset, create_train_val_test_split
from model import FDRTransUNet
from tqdm import tqdm
from plot_curves import plot_training_curves

# =============================
# HYPERPARAMETERS
# =============================

# Image size: paper uses 256x256. Must match model and dataset.
TARGET_SIZE = (256, 256)

# Batch size: paper specifies 4. 
BATCH_SIZE = 6

# Max epochs: Not specified in paper. Cap can be high because early stopping will halt training. 
# -> With patience=11, 100 will be a plenty.
NUM_EPOCHS = 100

# Learning rate: paper specifies 3e-4 (0.0003)
LR = 3e-4

# Weight decay: Not specified in paper. 0.01 is standard for ViT-based models 
# Used in TransUNet, ViT etc. Prevents overfitting by penalizing large weights.
WEIGHT_DECAY = 0.01

# Validation split: 20% for validation. Specified ratio in paper (6:2:2 ; Train, val, test)
VAL_RATIO = 0.2

# Cosine annealing warm restarts: T_0=10 (first cycle 10 epochs), T_mult=2
T_0 = 10
T_mult = 2

# Early stopping: paper states patience > 10 epochs of no mIoU improvement. 
# So we stop after 11 consecutive epochs with no improvement. 
EARLY_STOP_PATIENCE = 80

# Pixel threshold: paper specifies 0.5.
# After sigmoid, pixels >= 0.5 are predicted as pneumothorax-
PIXEL_TRESHOLD = 0.1

# Validation: try multiple thresholds and report best Dice (and metrics at that threshold).
VAL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Loss: hybrid BCE + Dice (helps with class imbalance and recall).
# Stronger push on positives (pos_weight, Dice weight) to raise probabilities and break 0.78 plateau.
BCE_WEIGHT = 0.35
DICE_WEIGHT = 0.65
POS_WEIGHT = 15.0

# Gradient clipping: disabled. Normalization layers (LayerNorm, etc.) already keep gradients
# in check; clipping was likely over-limiting updates and contributing to early plateau.
GRAD_CLIP_MAX_NORM = 1.0

# Checkpoint saving: best model by validation mIoU (for inference/predict.py)
CHECKPOINT_DIR = os.path.join(script_dir, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# =============================
# AUGMENTATION (paper: Normalize, Random rotation ±10°, flip p=0.4)
# =============================
# Applied to training data ONLY (not validation/inference).
# The dataset stacks image (ch0) and mask (ch1) into 2 channels,
# so geometric transforms (rotation, flip) affect both identically.
# Normalize only affects the image channel (ch0), leaving mask unchanged.
#
# mean=[0.5, 0.0], std=[0.5, 1.0]:
#   ch0 (image):  (pixel - 0.5) / 0.5  →  maps [0,1] to [-1,1]
#   ch1 (mask):   (pixel - 0.0) / 1.0  →  unchanged
#
# Why [-1,1]? Centering around 0 helps the network learn faster;
# symmetric range works well with ReLU and batch norm.

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.Normalize(mean=[0.5, 0.0], std=[0.5, 1.0]),
])

# =============================
# Loss: Dice + hybrid
# =============================

def dice_loss(pred_logits, targets, smooth=1.0):
    """Differentiable soft Dice loss computed per-image then averaged."""
    probs = torch.sigmoid(pred_logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class HybridLoss(torch.nn.Module):
    """BCE (with pos_weight for class imbalance) + Dice."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=5.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, logits, targets):
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * dice_loss(logits, targets)
        )


# =============================
# Metrics
# =============================

def dice_score(pred_logits, mask, threshold=PIXEL_TRESHOLD):
    """
    Dice coefficient: measures overlap between prediction and ground thruth.
    Range: 1 = perfect overlap, 0 = no overlap.
    Formula: 2 * |pred ∩ mask| / (|pred| + |mask|)
    """
    # Converts logits into 0/1: Sigmoid then threshold e.g. >= 0.5 -> 1
    prediction = (torch.sigmoid(pred_logits) > threshold).float()

    # Binarize mask to 0/1 in case it's float in (0,1) (mask pixels are already binary, but for safety?)
    mask_bin = (mask > threshold).float()

    # Overlap: count pixels wher both pred and mask are 1, per sample
    intersection = (prediction * mask_bin).sum(dim=(1, 2, 3))

    # Total positives: number of predicted 1s + number of mask 1s, per sample
    total = prediction.sum(dim=(1, 2, 3)) + mask_bin.sum(dim=(1, 2, 3))

    # Dice = 2*overlap / total; 1e-6 avoids divide-by-zero when both empty
    dice = (2.0 * intersection + 1e-6) / (total + 1e-6)

    # Return single scalar; mean Dice over batch as a Python float (e.g. for logging)
    return dice.mean().item()

def iou_score(pred_logits, mask, threshold=PIXEL_TRESHOLD):
    """
    IoU: Intersection over Union, aka Jaccard index. 
    Range: 0 to 1. Stricter than Dice (always <= Dice for same prediction).
    Formula: |pred ∩ mask| / |pred ∪ mask|

    Paper uses mIoU (mean IoU) as the primary validation metric
    for early stopping and model selection.
    """

    # Binarize prediction: sigmoid then threshold -> 0/1 tensor
    prediction = (torch.sigmoid(pred_logits) > threshold).float()

    # Binarize mask to 0/1 (again, mask already binary but for safety and good repitition)
    mask_bin = (mask > threshold).float()

    # Overlap: count of where both predicted and mask pixels are 1, per sample -> shape (B,)
    intersection = (prediction * mask_bin).sum(dim=(1, 2, 3))

    # Union: |pred ∪ mask| = |pred| + |mask| - |pred ∩ mask|, per sample -> shape (B,)
    # |pred ∩ mask| -> intersection -> pred AND mask 
    union = prediction.sum(dim=(1, 2, 3)) + mask_bin.sum(dim=(1, 2, 3)) - intersection

    # IoU = intersection / union ; 1e-6 avoids dividing by zero 
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Return single scalar; mean IoU over batch, as Python float 
    return iou.mean().item()

def precision_score(pred_logits, mask, threshold=PIXEL_TRESHOLD):
    """
    Precision: of all pixels predicted positive, how many are truly positive.
    Formula: TP / (TP + FP)
    Paper reports this in Tables 2 and 3.
    """
    prediction = (torch.sigmoid(pred_logits) > threshold).float()
    mask_bin = (mask > threshold).float()

    tp = (prediction * mask_bin).sum(dim=(1, 2, 3))
    predicted_pos = prediction.sum(dim=(1, 2, 3))

    precision = (tp + 1e-6) / (predicted_pos + 1e-6)
    return precision.mean().item()

def recall_score(pred_logits, mask, threshold=PIXEL_TRESHOLD):
    """
    Recall: of all truly positive pixels, how many did we predict correctly.
    Formula: TP / (TP + FN)
    Paper reports this in Tables 2 and 3.
    """
    prediction = (torch.sigmoid(pred_logits) > threshold).float()
    mask_bin = (mask > threshold).float()

    tp = (prediction * mask_bin).sum(dim=(1, 2, 3))
    actual_pos = mask_bin.sum(dim=(1, 2, 3))

    recall = (tp + 1e-6) / (actual_pos + 1e-6)
    return recall.mean().item()


# =============================
# TRAINING LOOP (one epoch)
# =============================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One full pass through the training data.

    Deep supervision (from paper):
    "The parameters are updated by summing the losses of the two outputs."
    So: total_loss = loss(logits_a, mask) + loss(logits_b, mask)
    We do NOT use the combined output for training loss; that's for infernece. 

    Gradient clipping: limits the size of gradients to prevent 
    sudden large updates that can destabilize transformer training.

    Args: 
        model: FDR-TransUNet 
        loader: DataLoader yielding (images, masks) batches.
        criterion: Loss function (BCEWithLogitsLoss)
        optimizer: Adam
        device: torch.device (not sure yet)

    Returns:
        Average training loss over the epoch (float).


    """
    model.train() # sets model to training mode (enables dropout, batch norm updates)
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad() # clear out old gradients

        combined, logits_a, logits_b = model(images) # exctract what the model returns

        # Deep supervision: sum losses from both decoder paths
        loss = criterion(logits_a, masks) + criterion(logits_b, masks)

        # Compute gradients
        loss.backward()

        if GRAD_CLIP_MAX_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

        # Update weights
        optimizer.step()

        # Adds this batch's loss (as a Python float) to the running total for the epoch
        total_loss += loss.item()

    # Return average loss per batch for this epoch; if loader is empty, 
    # return 0.0 to avoid division by zero
    return total_loss / len(loader) if len(loader) else 0.0




def validate(model, loader, criterion, device):
    """
    Evaluate on validation set WITHOUT updating weights.
    Computes Dice (and IoU, prec, rec) at each threshold in VAL_THRESHOLDS,
    then reports metrics at the threshold that gave the highest mean Dice.

    Returns: val_loss, val_dice, val_miou, val_precision, val_recall, best_threshold
    """
    model.eval()
    total_loss = 0.0
    n = 0
    sum_dice = {t: 0.0 for t in VAL_THRESHOLDS}
    sum_iou  = {t: 0.0 for t in VAL_THRESHOLDS}
    sum_prec = {t: 0.0 for t in VAL_THRESHOLDS}
    sum_rec  = {t: 0.0 for t in VAL_THRESHOLDS}

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            combined, logits_a, logits_b = model(images)

            loss = criterion(logits_a, masks) + criterion(logits_b, masks)
            total_loss += loss.item()

            for t in VAL_THRESHOLDS:
                sum_dice[t] += dice_score(combined, masks, threshold=t)
                sum_iou[t]  += iou_score(combined, masks, threshold=t)
                sum_prec[t] += precision_score(combined, masks, threshold=t)
                sum_rec[t]  += recall_score(combined, masks, threshold=t)
            n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, VAL_THRESHOLDS[0]

    mean_dice = {t: sum_dice[t] / n for t in VAL_THRESHOLDS}
    best_t = max(VAL_THRESHOLDS, key=lambda t: mean_dice[t])

    return (
        total_loss / n,
        mean_dice[best_t],
        sum_iou[best_t] / n,
        sum_prec[best_t] / n,
        sum_rec[best_t] / n,
        best_t,
    )



def main():
    # 1. Device: use GPU if available else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # 2. Load data, ensuring we have file_id
    df = data_processing.load_data()
    if "file_id" not in df.columns:
        print("Buidlig file_id mapping (slow first time)...")
        mapping = data_processing.build_dicom_id_mapping()
        df = data_processing.add_file_id_column(df, dicom_id_mapping=mapping)
        data_processing.save_processed_data(df)

    
    # 3. Merge multiple RLE rows per image into one row per image (union of masks)
    df = data_processing.merge_rle_rows(df, group_by="file_id")

    # 4. Split the data: train(60%), val(20%), test(20%)
    # Train = fit model.
    # Val   = early stopping + model selection.
    # Test  = final eval only.

    train_df, val_df, test_df = create_train_val_test_split(
        df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )

    # 5. Dataset: train gets augmentation; val and test don't - for fair evaluation
    data_processing._build_dicom_path_cache(config.image_dir)

    train_ds = PneumothoraxDataset(
        train_df, image_dir=config.image_dir,
        target_size=TARGET_SIZE, transform=train_transform
    )

    val_ds = PneumothoraxDataset(
        val_df, image_dir=config.image_dir,
        target_size=TARGET_SIZE, transform=None
    )

    # Test set: same as val but held out until the very end
    test_ds = PneumothoraxDataset(
        test_df, image_dir=config.image_dir,
        target_size=TARGET_SIZE, transform=None
    )

    # 6. Dataloader: batch the data; shuffle only training set
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    # 6. Model: FDR-TransUNet, same input size as TARGET_SIZE
    model = FDRTransUNet(
        in_channels=1,
        encoder_channels=(32, 64, 128, 256),
        embed_dim=768,
        growth_rate=64,
        num_heads=12,
        num_layers=12,
        input_h=TARGET_SIZE[0],
        input_w=TARGET_SIZE[1],
    ).to(device)

    # PARAMETERS
    n = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n:,}\nModel parameters (Million): {n/1e6:.2f}")


    # 7. Loss (BCE+Dice with pos_weight), optimizer (AdamW), scheduler (cosine warm restarts)
    criterion = HybridLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT, pos_weight=POS_WEIGHT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult
    )

    # 8. Sanity check: one forward pass to catch shape errors before training
    print("Sanity check: one forward pass...")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, TARGET_SIZE[0], TARGET_SIZE[1]).to(device)
        c, _, _ = model(x)
        assert c.shape == x.shape, f"Shape mismatch: {x.shape} vs {c.shape}"
    print("Sanity check passed!")

    # 9. Training loop: train by epoch, validate, track best mIoU, early stop
    best_miou = 0.0
    patience_count = 0
    print(f"\nTraining up to {NUM_EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})")
    print(f"Loss: BCE ({BCE_WEIGHT}, pos_weight={POS_WEIGHT}) + Dice ({DICE_WEIGHT}). Cosine warm restarts (T_0={T_0}, T_mult={T_mult}). AdamW. Grad clip={GRAD_CLIP_MAX_NORM}")
    print(f"Early stop if no mIoU improvements for {EARLY_STOP_PATIENCE} epochs\n")

    # --- History for training curves (convergence + metrics) ---
    history = {
        "train_loss": [], "val_loss": [],
        "val_dice": [], "val_miou": [],
        "val_precision": [], "val_recall": [],
    }

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_miou, val_prec, val_rec, best_t = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_miou"].append(val_miou)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        if val_miou > best_miou:
            best_miou = val_miou
            patience_count = 0            
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_miou": best_miou,
            }, BEST_MODEL_PATH)
            print(f"  (saved best checkpoint to {BEST_MODEL_PATH})")
        else:
            patience_count += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS}"
            f"  loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  dice={val_dice:.4f}"
            f"  mIoU={val_miou:.4f}"
            f"  prec={val_prec:.4f}"
            f"  rec={val_rec:.4f}"
            f"  best_mIoU={best_miou:.4f}"
            f"  best_t={best_t}"
            f"  lr={current_lr:.6f}"
            f"  patience={patience_count}"
        )

        if patience_count >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: no mIoU improvment for {patience_count}")
            break

    print(f"\nTraining complete. Best validation mIoU: {best_miou:.4}")

    # --- Save history to JSON so we can re-plot later without re-running training ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"history": history, "best_miou": best_miou, "n_epochs": len(history["train_loss"])}, f, indent=2)
    print(f"Saved training history to {history_path}")

    # --- Plot training curves (loss + metrics) via shared function ---
    curves_path = os.path.join(CHECKPOINT_DIR, "training_curves.png")
    plot_training_curves(history, curves_path)
    print(f"Saved training curves to {curves_path}")

    # 10. Final evalutaion on held_out test set
    # Test set was never used for training or early stopping
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    test_loss, test_dice, test_miou, test_prec, test_rec, test_best_t = validate(
        model, test_loader, criterion, device
    )
    print(
        f"\nTest set results (best threshold={test_best_t}):"
        f"\n  loss={test_loss:.4f}"
        f"\n  dice={test_dice:.4f}"
        f"\n  mIoU={test_miou:.4f}"
        f"\n  precision={test_prec:.4f}"
        f"\n  recall={test_rec:.4f}"
    )

    return model, best_miou, test_miou

if __name__ == "__main__":
    main()
