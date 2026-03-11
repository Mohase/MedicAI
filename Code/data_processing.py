# Data Processing Functions - Loading, Mapping and processing

import pandas as pd
import pydicom
import os 
import socket
from tqdm import tqdm
import config

def load_data(processed_path=None, original_path=None):
    """Load DataFrame, preferring processed version"""
    if processed_path is None:
        processed_path = config.processed_csv
    if original_path is None:
        original_path = config.original_csv

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        print("Loaded processed CSV with file_id column")
    else:
        df = pd.read_csv(original_path)
        print("Loaded original CSV")

    df.columns = df.columns.str.strip()
        
    return df



def get_all_dcm_files(image_directory=None):
    """Get all DICOM file paths recursively"""
    if image_directory is None:
        image_directory = config.image_dir

    dcm_files = []
    for root, _, files in os.walk(image_directory):
        for file in files:
            if file.endswith('.dcm') and not file.startswith('._'):
                dcm_files.append(os.path.join(root, file))
    return dcm_files




def build_dicom_id_mapping(image_directory=None, dcm_files=None):
    """Build mapping dictionary from ImageId (long UID) to file_id (short UID)"""
    if image_directory is None:
        image_directory = config.image_dir

    if dcm_files is None:
        dcm_files = get_all_dcm_files(image_directory)
    
    dicom_id_mapping = {}
    print("Building DICOM ID mapping... This might take a moment.")

    for filepath in tqdm(dcm_files, desc="Processing DICOM files"):
        try: 
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            long_id = dcm.file_meta.MediaStorageSOPInstanceUID
            short_id = dcm.SOPInstanceUID
            dicom_id_mapping[long_id] = short_id
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
        
    return dicom_id_mapping

def add_file_id_column(df, dicom_id_mapping=None, image_directory=None):
    """Add file_id column to DataFrame using DICOM ID mapping"""
    if 'file_id' in df.columns:
        print("file_id column already exists")
        return df

    if dicom_id_mapping is None:
        dicom_id_mapping = build_dicom_id_mapping(image_directory)

    df['file_id'] = df['ImageId'].map(dicom_id_mapping)
    print("Added file_id column to DataFrame")

    # Check for unmatched IDs
    unmatched_ids = df[df['file_id'].isna()]['ImageId'].unique()
    if len(unmatched_ids) > 0:
        print(f"Warning: {len(unmatched_ids)} ImageIds did not find a match in DICOM files.")
    else: 
        print("No unmatched ImageIds")

    return df

def merge_rle_rows(df, group_by="file_id"):
    """
    Merge multiple CSV rows per image into one row per image.
    SIIM can have several EncodedPixels per ImageId (e.g. one per lung).
    Returns a DataFrame with one row per image and column EncodedPixels_list
    (list of RLE strings for that image, excluding -1/empty). Use this df for
    training so each image is trained once with the union of all RLE masks.
    """
    if group_by not in df.columns:
        raise ValueError(f"merge_rle_rows: column '{group_by}' not in df")
    id_col = "ImageId" if group_by == "file_id" else "file_id"
    if id_col in df.columns:
        first_id = df.groupby(group_by)[id_col].first().reset_index()
    rows = []
    for key, group in df.groupby(group_by):
        rles = group["EncodedPixels"].dropna().astype(str).tolist()
        rles = [r for r in rles if r != "-1" and r.strip()]
        other = {group_by: key}
        if id_col in df.columns:
            other[id_col] = group[id_col].iloc[0]
        other["EncodedPixels_list"] = rles
        rows.append(other)
    merged = pd.DataFrame(rows)
    n_before = len(df)
    n_after = len(merged)
    print(f"Merged RLE rows: {n_before} -> {n_after} (one row per image, {n_before - n_after} duplicate rows removed)")
    return merged


def save_processed_data(df, csv_path=None, parquet_path=None):
    """Save processed DataFrame to CSV and Parquet"""
    if csv_path is None:
        csv_path = config.processed_csv
    if parquet_path is None:
        parquet_path = config.processed_parquet

    if 'file_id' not in df.columns:
        print("WARNING: file_id column not found! Cannot save processed data.")
        return 

    # Save as CSV
    df.to_csv(csv_path, index=False)
    print(f"DataFrame saved to CSV: {csv_path}")
    
    # Save as Parquet
    try: 
        df.to_parquet(parquet_path, index=False)
        print(f"DataFrame saved to Parquet: {parquet_path}")
    except Exception as e:
        print(f"Could not save Parquet file: {e}")
    

_dicom_path_cache = None

def _build_dicom_path_cache(image_directory=None):
    """Walk the DICOM tree once and build file_id -> path dict."""
    global _dicom_path_cache
    if image_directory is None:
        image_directory = config.image_dir
    print("Building DICOM path cache...")
    _dicom_path_cache = {}
    for root, _, files in os.walk(image_directory):
        for file in files:
            if file.endswith('.dcm') and not file.startswith('._'):
                filepath = os.path.join(root, file)
                file_id = file.replace('.dcm', '')
                _dicom_path_cache[file_id] = filepath
    print(f"Cached {len(_dicom_path_cache)} DICOM paths")
    return _dicom_path_cache

def find_dicom_file(file_id, image_directory=None):
    """Find DICOM file path by file_id. Uses cache for O(1) lookup."""
    global _dicom_path_cache
    if _dicom_path_cache is None:
        _build_dicom_path_cache(image_directory)
    if file_id in _dicom_path_cache:
        return _dicom_path_cache[file_id]
    for fid, path in _dicom_path_cache.items():
        if file_id in path:
            return path
    return None

def load_dicom_image(file_path):
    """Load DICOM image and return pixel array and DICOM object.

    Uses force=True so files missing the DICOM File Meta header (e.g. some SIIM
    files) can still be read. When file_meta is incomplete (no Transfer Syntax UID),
    sets a default so pixel data can be decoded. Wraps pydicom.dcmread so that I/O
    timeouts are surfaced as TimeoutError, allowing the dataset to skip problematic files.
    """
    try:
        dicom = pydicom.dcmread(file_path, force=True)
    except TimeoutError as e:
        raise TimeoutError(f"Timeout reading DICOM: {file_path}") from e
    except (OSError, socket.timeout) as e:
        raise TimeoutError(f"I/O error reading DICOM: {file_path}") from e

    # force=True can leave file_meta without Transfer Syntax UID, so pixel_array fails.
    # Set a default (Implicit VR Little Endian) so decoding can proceed.
    try:
        from pydicom.uid import ImplicitVRLittleEndian
        file_meta = getattr(dicom, "file_meta", None)
        if file_meta is not None and not getattr(file_meta, "TransferSyntaxUID", None):
            file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    except Exception:
        pass

    return dicom.pixel_array, dicom



