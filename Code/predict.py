"""
Predict pneumothorax segmentation from a single image.
Saves probabilty mask, binary mask, overlay image, and scores.

Usage: 
    python3 Code/predict.py path/to/image.dcm
    python3 Code/predict.py path/to/image.png
    (optional) python Code/predict.py path/to/image.dcm outputs/subdir

From code: 
    from predict import load_model, predict_from_path, save_overlay_and_mask
    model = load_model()
    out = predict_from_path(model, "path/to/image.dcm")
    save_overlay_and_mask("path/to/image.dcm", out, save_dir="outputs") 
"""

import os
import sys
import numpy as np
import torch

# Path setup: so we can run as "python Code/predict.py" from project root (Code)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from skimage.transform import resize
import config
import data_processing
from model import FDRTransUNet

# Must match training (model expects 256x256 input)
TARGET_SIZE = (256, 256)
CHECKPOINT_DIR = os.path.join(script_dir, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
DEFAULT_SAVE_DIR = os.path.join(script_dir, "outputs")

# Folder for test images; only the filename is given on CLI (or use default below).
TEST_IMAGES_DIR = os.path.abspath(os.path.join(script_dir, "..", "images"))
# Default filename when no argument is passed (change this or pass filename on CLI).
DEFAULT_TEST_IMAGE = "test.dcm"

def load_model():
    """ Load trained FDR-TransUNet from best checkpoint."""
    # Build same architechure as in train.py (weights come from file)
    model = FDRTransUNet(
        in_channels=1,
        encoder_channels=(32, 64, 128, 256),
        embed_dim=768,
        growth_rate=64,
        num_heads=12,
        num_layers=12,
        input_h=TARGET_SIZE[0],
        input_w=TARGET_SIZE[1],
    )

    if os.path.exists(BEST_MODEL_PATH):
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        # Supports both old (bare state_dict) and new (dict with metadata) formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"No checkpoint at {BEST_MODEL_PATH}. Train first.")
    model.to(DEVICE)
    model.eval() # We use for infernece, so disables dropout etc.
    return model

def preprocess_image(image):
    """
    image: numpy array (H, W) or (H, W, C), range 0-255 or 0-1.
    Same as dataset: 0-1, resize, then (x-0.5)/0.5 to match training normalize. 
    Returns: tensor (1, 1, 256, 256) on DEVICE
    """
    # Normazlize to 0-1 if image is 0-255
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    # Grayscale if RGB, take mean of channels
    if image.ndim==3:
        image = image.mean(axis=2)
    
    image = resize(image, TARGET_SIZE, preserve_range=True, anti_aliasing=True)

    # Match training: Normalize(mean=[0.5], std=[0.5]) -> (x - 0.5) / 0.5
    image = (image - 0.5) / 0.5

    # Add batch and channel dims: (256,256) -> (1, 1, 256, 256)
    image = np.expand_dims(image, axis=(0, 1))
    t = torch.from_numpy(image).float().to(DEVICE)
    return t 

def predict_from_image(model, image):
    """
    Run model on one image (numpy array).
    Returns dict:
        prob_mask: (256, 256) float [0,1]
        binary_mask: (256, 256) 0 or 1 at 0.5 threshold 
        confidence: float (mean prob, in predicted positive region, or max prob in none)
        has_pneumothorax: bool
        fraction_positive: float
    """
    x = preprocess_image(image)
    with torch.no_grad(): # no gradients needed. Saves memory and time
        combined, _, _ = model(x)
    
    # Get logits for the single image, move to CPU, convert to numpy.
    # numpy works only on cpu
    logits = combined[0, 0].cpu().numpy()
    # sigmoid: turn logits into probabilities in [0, 1]
    prob_mask = 1.0  / (1.0 + np.exp(-logits))

    # Binary mask: 1 where model says pnemou (prob >= 0.5), 0 elsewhere
    binary_mask = (prob_mask >= 0.5).astype(np.float32)
    has_pneumo = binary_mask.sum() > 0 # again, stores bool; if even 1 pixel 1 -> has pneumo

    # What fraction of the image is predicted as pneumo
    fraction_positive = float(binary_mask.sum() / binary_mask.size)

    # Confidence: average probabilty in the predicted positive region (or max prob if no positive pixels)
    # Confidence = mean prob in the region we predicted as pneumo (not over all pixels).
    # Edge case: one pixel at 0.9 gives 0.9; with a proper model this is rare—usually we get a coherent blob or nothing.
    if has_pneumo:
        confidence = float(prob_mask[binary_mask > 0.5]. mean())
    else:
        confidence = float(prob_mask.max())

    return {
        "prob_mask": prob_mask, 
        "binary_mask": binary_mask,
        "confidence": confidence,
        "has_pneumothorax": has_pneumo,
        "fraction_positive": fraction_positive,
    }


def predict_from_path(model, path):
    """ Load image from path (DICOM or image file), then predict."""
    path = os.path.abspath(path)
    if path.lower().endswith(".dcm"):
        image, _ = data_processing.load_dicom_image(path)
    else: 
        try:
            image = np.load(path)
        except Exception:
            # We use PIL here (and not elsewhere) because predict_from_path must accept any path:
            # DICOM, .npy, or a regular image file (jpg/png). The dataset only loads DICOM for SIIM,
            # so it doesn't need PIL; this function is a flexible entry point for inference.
            from PIL import Image
            image = np.array(Image.open(path).convert("L"))
    return predict_from_image(model, image)


def save_overlay_and_mask(image_path_or_array, result, save_dir=None, base_name=None):
    """
    Save: 
        - <base>_mask.png: grayscale probability map (0-255).
        - <base>_overlay.png: original (resized) with red tint where pneumo predicted + text.
        - <base>_scores.txt: has_pneumothorax, confidence, fraction_positive

    result: dict from predict_from_image/predict_from_path.
    image_path_or_array : path to image (used to load for overlay) or numpy array (H,W).
    save_dir: folder to write into (default: Code/outputs).
    base_name: filename without extension (default: from path or "prediction").
    """
    if save_dir is None:
        save_dir =  DEFAULT_SAVE_DIR
    # Create the folder if it doesn't exist; if it already exists, do nothing (exist_ok=True). 
    os.makedirs(save_dir, exist_ok=True)

    # Decide output filenames: from image path or "prediction"
    if base_name is None:
        if isinstance(image_path_or_array, str):
            base_name = os.path.splitext(os.path.basename(image_path_or_array))[0]
        else:
            base_name = "prediction"

    prob_mask = result["prob_mask"]
    binary_mask = result["binary_mask"]
    confidence = result["confidence"]
    has_pneumo = result["has_pneumothorax"]
    fraction_positive = result["fraction_positive"]

    from skimage.transform import resize
    if isinstance(image_path_or_array, str):
        if image_path_or_array.lower().endswith(".dcm"):
            img, _ = data_processing.load_dicom_image(image_path_or_array)
        else:
            try:
                img = np.load(image_path_or_array)
            except Exception:
                from PIL import Image
                img = np.array(Image.open(image_path_or_array).convert("L"))
    else:
        img = np.asarray(image_path_or_array)
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = resize(img, TARGET_SIZE, preserve_range=True, anti_aliasing=True)
    img = np.clip(img, 0, 1)

    # Save probabilty mask: 0-255 grayscale PNG
    mask_uint8 = (np.clip(prob_mask, 0, 1) * 255).astype(np.uint8)
    mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
    from PIL import Image
    Image.fromarray(mask_uint8).save(mask_path)
    print(f"Saved mask: {mask_path}")

    # Overlay: grayscale image + red where binary_mask is 1
    overlay = np.stack([img, img, img], axis=-1) # (H,W,3) grayscale
    # Appends a color channel to the "place holder" img
    red_tint= np.stack([binary_mask, np.zeros_like(binary_mask), np.zeros_like(binary_mask)], axis=-1) 
    overlay = np.clip(overlay + 0.5 * red_tint, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_uint8)

    # Summary text on image

    try: 
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(overlay_pil)
        text = f"Pneumothorax: {'Yes' if has_pneumo else 'No'}\nConfidence: {confidence:.2f}\nArea: {fraction_positive:.2%}"
        draw.text((10, 10), text, fill=(255, 255, 255))
    except Exception:
        pass
    overlay_path = os.path.join(save_dir, f"{base_name}_overlay.png")
    overlay_pil.save(overlay_path)
    print(f"Saved overlay: {overlay_path}")

    # Scores as plain text file
    scores_path = os.path.join(save_dir, f"{base_name}_scores.txt")
    with open(scores_path, "w") as f:
        f.write(f"has_pneumothorax= {has_pneumo}\n")
        f.write(f"confidence= {confidence:.4}\n")
        f.write(f"fraction_positive= {fraction_positive:.4f}\n")
    print(f"Saved scores to: {scores_path}")


def main():
    # Filename: from CLI or default. Output dir: optional second argument.
    filename = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_TEST_IMAGE
    save_dir = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_SAVE_DIR  

    # Full path = test images folder + filename
    path = os.path.join(TEST_IMAGES_DIR, filename)

    print(f"Loading model from {BEST_MODEL_PATH}...")
    model = load_model()
    print(f"Loading image: {path}...")
    out = predict_from_path(model, path)
    print(f"Has pneumothorax:   {out['has_pneumothorax']}")
    print(f"Confidence:         {out['confidence']:.4f}")
    print(f"Fraction positive:  {out['fraction_positive']:.4f}")
    save_overlay_and_mask(path, out, save_dir=save_dir)
    return out


if __name__ == "__main__":
    main()

