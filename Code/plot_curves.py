"""
Re-plot training curves from saved history (no need to re-run training).

Usage:
    python Code/plot_curves.py
    python Code/plot_curves.py path/to/training_history.json

Reads training_history.json from Code/checkpoints/ (or the path you give),
draws loss + metrics curves, and saves training_curves.png in the same folder.

Also provides plot_training_curves(history, save_path) for use from train.py.
"""
import os
import sys
import json
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(script_dir, "checkpoints")
DEFAULT_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "training_history.json")


def plot_training_curves(history, save_path, dpi=150):
    """
    Draw loss + validation metrics and save to save_path.

    history: dict with keys train_loss, val_loss, val_dice, val_miou (lists).
    save_path: path for the PNG file (e.g. checkpoints/training_curves.png).
    """
    n = len(history["train_loss"])
    epochs_x = range(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_x, history["train_loss"], label="Train loss", color="C0")
    axes[0].plot(epochs_x, history["val_loss"], label="Val loss", color="C1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training convergence (loss)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs_x, history["val_dice"], label="Val Dice", color="C2")
    axes[1].plot(epochs_x, history["val_miou"], label="Val mIoU", color="C3")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    history_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HISTORY_PATH
    if not os.path.exists(history_path):
        print(f"Not found: {history_path}")
        print("Run training first so it saves training_history.json")
        sys.exit(1)

    with open(history_path) as f:
        data = json.load(f)
    history = data["history"]
    best_miou = data.get("best_miou", None)

    out_dir = os.path.dirname(history_path)
    curves_path = os.path.join(out_dir, "training_curves.png")
    plot_training_curves(history, curves_path)
    print(f"Saved {curves_path}" + (f" (best_mIoU={best_miou:.4f})" if best_miou is not None else ""))


if __name__ == "__main__":
    main()
