"""
PneumothoraxDataset

This file defines a custom PyTorch Dataset for the SIIM Pneumothorax Detection task.

Purpose:
- Read image-level labels from a CSV file
- Load corresponding DICOM chest X-ray images from disk
- Convert run-length encoded (RLE) annotations into pixel-wise segmentation masks
- Return (image, mask) pairs that can be used to train a segmentation model
  such as FDR-TransUNet

High-level workflow:
1. Each row in the CSV represents one chest X-ray image
2. The dataset locates the matching DICOM file using the file_id
3. The DICOM image is loaded and converted into a NumPy array
4. The EncodedPixels field is examined:
   - If it is '-1' or missing, the image has no pneumothorax and an empty mask is created
   - Otherwise, the RLE string is decoded into a binary segmentation mask
5. The image and mask are returned as a single training sample

Why this is needed:
- Deep learning models cannot read DICOM files or CSV annotations directly
- This dataset class acts as a bridge between raw medical data and the model
- It ensures that every image is paired with the correct ground-truth mask

Medical context:
- Pneumothorax may or may not be present in each image
- Empty masks are valid and important, as the model must learn what healthy lungs look like
- Accurate mask generation is critical for reliable medical image segmentation

What each step in __getitem__ does:
1: Gets one row from your DataFrame
2: Finds and loads the DICOM file
3: Converts RLE to mask (handles -1 = no pneumothorax)
4: Normalizes pixel values from 0-255 to 0-1 (convert to float)
5: Resizes if needed (1024→512) - optional
6: Adds channel dimension (H, W) -> (1, H, W) for grayscale
7: Converts numpy arrays to PyTorch tensors
8: Applies data augmentation if provided
Returns: (image, mask) tuple as PyTorch tensors

"""

import torch 
from torch.utils.data import Dataset
import numpy as np
import pandas as pd 
from torchvision import transforms 
import data_processing
import mask_functions as ms
import config

class PneumothoraxDataset(Dataset):
    def __init__(self, df, image_dir=None, target_size=(512,512), transform=None):
        """
        Args: 
            df: Dataframe with ImageId, EncodedPixels, file_id columns
            image_dir: Directory with DICOM images
            target_size: Target image size (height, width) - optional resize
            transform: Optional torchvision transforms for augmentation 
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir if image_dir else config.image_dir
        self.target_size = target_size
        self.transform = transform 
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.df)

    def __getitem__(self, idx):
        """Load and return one image-mask pair"""
        # 1. Get the row from DataFrame
        row = self.df.iloc[idx]

        # 2. Find and load the DICOM image
        dicom_path = data_processing.find_dicom_file(row['file_id'], self.image_dir)
        if dicom_path is None: 
            raise FileNotFoundError(f"DICOM file not found for file_id: {row['file_id']}")

        try:
            image_array, _ = data_processing.load_dicom_image(dicom_path)
        except TimeoutError:
            print(f"Warning: timeout reading DICOM {dicom_path}, skipping this sample.")
            new_idx = (idx + 1) % len(self.df)
            return self.__getitem__(new_idx)
        except AttributeError as e:
            # Missing Transfer Syntax UID or other pixel decode failure (e.g. on glass).
            print(f"Warning: cannot decode pixel data for {dicom_path}: {e}, skipping this sample.")
            new_idx = (idx + 1) % len(self.df)
            return self.__getitem__(new_idx)

        # 3. Get the rle mask from string
        rle_string = row['EncodedPixels']
        if rle_string == '-1' or pd.isna(rle_string):
            # No pneumothorax -> create an empty mask
            mask = np.zeros(image_array.shape, dtype=np.uint8)
        else:
            height, width = image_array.shape
            mask = ms.rle2mask(rle_string, width, height).T

        # 4. Normalize to 0-1 range (convert to float)
        image = image_array.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # 5. Optional: Resize if target_size is different 
        if self.target_size and image_array.shape != self.target_size:
            from skimage.transform import resize
            image = resize(image, self.target_size, preserve_range=True, anti_aliasing=True)
            mask = resize(mask, self.target_size, preserve_range=True, anti_aliasing=False, order=0)

        # 6. Add channel dimension (H, W) -> (1, H, W) for grayscale
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # 7. Convert to PyTorch tensors 
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # 8. Apply transforms if provided (for augmentation)
        if self.transform:
            # Stack image and mask together for joint transformation
            stacked = torch.cat([image, mask], dim=0)
            stacked = self.transform(stacked)
            image = stacked[0:1]
            mask = stacked[1:2]

        return image, mask 
    
def create_train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    """
    Split DataFrame into train(60%), validation(20%) and test(20%) sets.
    Validation is used for early stopping and model selection;
        test is held out for final evaluation only.
    
    """
    # Ensure correct ratios and allow for tiny float error
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Shuffle rows so train/val/test are random; fixed seed for reproducibility
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    n = len(df_shuffled)

    # Number of rows for each split (train and val use ratios; test gets rest)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # Calculate split point 
    val_size = int(len(df_shuffled) * val_ratio)

    # Split into train, val and test in contiguous chunks: [0:n_train], [n_train:n_train+n_val], [n_train+n_val:]
    train_df = df_shuffled[:n_train]
    val_df   = df_shuffled[n_train:n_train + n_val]
    test_df  = df_shuffled[n_train + n_val:]

    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")


    return train_df, val_df, test_df