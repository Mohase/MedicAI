"""
Test whether the SIIM CSV has multiple rows per image (multiple RLEs per ImageId).

If many images have 2+ rows, we should merge RLEs into one mask per image for training.
Run from project root: python Code/test_rle_per_image.py
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
import pandas as pd

def main():
    # Load CSV (prefer processed, fallback to original)
    processed_path = config.processed_csv
    original_path = config.original_csv

    if os.path.exists(processed_path):
        path = processed_path
        df = pd.read_csv(path)
        print(f"Loaded: {path}")
    elif os.path.exists(original_path):
        path = original_path
        df = pd.read_csv(path)
        print(f"Loaded: {path}")
    else:
        print(f"Neither found: {processed_path} or {original_path}")
        sys.exit(1)

    df.columns = df.columns.str.strip()

    # Rows and unique images by ImageId
    n_rows = len(df)
    n_unique_image_id = df["ImageId"].nunique()
    n_unique_file_id = df["file_id"].nunique() if "file_id" in df.columns else None

    print(f"\n--- Rows vs unique images ---")
    print(f"Total rows in CSV:     {n_rows}")
    print(f"Unique ImageId:        {n_unique_image_id}")
    if n_unique_file_id is not None:
        print(f"Unique file_id:        {n_unique_file_id}")

    # Rows per image (by ImageId)
    counts_per_image = df.groupby("ImageId").size()

    print(f"\n--- Rows per ImageId ---")
    one = (counts_per_image == 1).sum()
    two = (counts_per_image == 2).sum()
    three_plus = (counts_per_image >= 3).sum()
    max_rows = counts_per_image.max()

    print(f"Images with exactly 1 row:  {one} ({100*one/len(counts_per_image):.1f}%)")
    print(f"Images with exactly 2 rows: {two} ({100*two/len(counts_per_image):.1f}%)")
    print(f"Images with 3+ rows:        {three_plus} ({100*three_plus/len(counts_per_image):.1f}%)")
    print(f"Max rows for one image:     {max_rows}")

    if n_rows > n_unique_image_id:
        print(f"\n>>> Multiple rows per image: YES. {n_rows - n_unique_image_id} extra rows (same image, different RLE).")
        print("    Training currently uses each row separately; consider merging RLEs per image.")
    else:
        print(f"\n>>> Multiple rows per image: NO. One row per ImageId.")

    # Show a few examples of images with multiple rows
    multi = counts_per_image[counts_per_image >= 2]
    if len(multi) > 0:
        print(f"\n--- Example ImageIds with multiple RLEs (first 5) ---")
        for i, (image_id, count) in enumerate(multi.head(5).items()):
            rows = df[df["ImageId"] == image_id]
            rles = rows["EncodedPixels"].tolist()
            has_neg = any(r == "-1" or (isinstance(r, float) and pd.isna(r)) for r in rles)
            print(f"  {image_id[:50]}... -> {count} rows, has -1/empty: {has_neg}")

    # Same check by file_id if present
    if "file_id" in df.columns:
        counts_per_file = df.groupby("file_id").size()
        n_multi_file = (counts_per_file >= 2).sum()
        print(f"\n--- By file_id ---")
        print(f"file_ids with 2+ rows: {n_multi_file}")

if __name__ == "__main__":
    main()
