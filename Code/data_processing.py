# Data Processing Functions - Loading, Mapping and processing

import pandas as pd
import pydicom
import os 
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
            if file.endswith('.dcm'):
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
    

def find_dicom_file(file_id, image_directory=None):
    """Find DICOM file path by file_id"""
    if image_directory is None:
        image_directory = config.image_dir
    
    for root, _, files in os.walk(image_directory):
        for file in files:
            if file.endswith('.dcm'):
                filepath = os.path.join(root, file)
                if file_id in filepath or os.path.basename(filepath).replace('.dcm', '') == file_id:
                    return filepath
    return None

def load_dicom_image(file_path):
    """Load DICOM image and return pixel array and DICOM object"""
    dicom = pydicom.dcmread(file_path)
    return dicom.pixel_array, dicom



