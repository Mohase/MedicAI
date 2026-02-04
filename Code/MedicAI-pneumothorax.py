# MedicAI Pneumothorax Detection - Main Workflow
# %%
# Imports
import sys
import os

# Add Code directory to Python path so imports work
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our modules
import config
import data_processing
import visualization

# %%
# Load Data
df = data_processing.load_data()
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
df.head()

# %%
# Optional: Build mapping and add file_id if it doesn't exist
# (Only needed if file_id column is missing)
if 'file_id' not in df.columns:
    print("file_id column not found. Building mapping...")
    df = data_processing.add_file_id_column(df)
    data_processing.save_processed_data(df)
    print("Mapping complete and saved!")
else:
    print("file_id column already exists")

# %%
# Get a sample image with pneumothorax
pneumo_row = df[df['EncodedPixels'] != '-1'].iloc[0]
pneumo_row2 = df[df['EncodedPixels'] != '-1'].iloc[1]
print(f"Sample Image ID: {pneumo_row['ImageId']}")
print(f"File ID: {pneumo_row['file_id']}")

# %%
# Load and visualize pneumothorax
dicom_path = data_processing.find_dicom_file(pneumo_row['file_id'])
dicom_path2 = data_processing.find_dicom_file(pneumo_row2['file_id'])


if dicom_path:
    print(f"Found DICOM file: {dicom_path}")
    
    # Load the DICOM image
    image_array, dicom_obj = data_processing.load_dicom_image(dicom_path)
    print(f"Image shape: {image_array.shape}")
    
    # Get the mask
    mask = visualization.get_mask_for_image(pneumo_row['EncodedPixels'], image_array.shape)
    print(f"Mask shape: {mask.shape}")
    
    # Visualize
    visualization.visualize_pneumothorax(image_array, mask, pneumo_row['ImageId'])
else:
    print("DICOM file not found")

if dicom_path:
    print(f"Found DICOM file: {dicom_path2}")
    
    # Load the DICOM image
    image_array, dicom_obj = data_processing.load_dicom_image(dicom_path2)
    print(f"Image shape: {image_array.shape}")
    
    # Get the mask
    mask = visualization.get_mask_for_image(pneumo_row2['EncodedPixels'], image_array.shape)
    print(f"Mask shape: {mask.shape}")
    
    # Visualize
    visualization.visualize_pneumothorax(image_array, mask, pneumo_row2['ImageId'])
else:
    print("DICOM file not found")

# %%
# Check image sizes across a larger random sample
from collections import Counter
import random

print("Checking image sizes across random sample...")
sample_size = 500  # Check 500 random images
image_sizes = []

# Get random indices
indices = random.sample(range(len(df)), min(sample_size, len(df)))

for idx in indices:
    row = df.iloc[idx]
    dicom_path = data_processing.find_dicom_file(row['file_id'])
    if dicom_path:
        try:
            image_array, dicom_obj = data_processing.load_dicom_image(dicom_path)
            image_sizes.append(image_array.shape)
        except Exception as e:
            print(f"Error loading image {idx}: {e}")

if image_sizes:
    unique_sizes = set(image_sizes)
    print(f"\nChecked {len(image_sizes)} images")
    print(f"Unique image sizes found: {len(unique_sizes)}")
    print(f"Sizes: {unique_sizes}")
    
    if len(unique_sizes) == 1:
        print("✓ All sampled images are the same size!")
    else:
        print("⚠ Images have different sizes - will need resizing in dataset")
        
    # Show size distribution
    size_counts = Counter(image_sizes)
    print("\nSize distribution:")
    for size, count in size_counts.most_common():
        print(f"  {size}: {count} images ({count/len(image_sizes)*100:.1f}%)")

# %%