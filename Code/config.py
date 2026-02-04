# Config - Setting up paths 

import os 

# Get the directory where this config file is located
config_dir = os.path.dirname(os.path.abspath(__file__))
""" /Users/mohammadasender/Desktop/medicAI/Code """

project_root = os.path.dirname(config_dir)
""" /Users/mohammadasender/Desktop/medicAI """

# Data paths

data_path = os.path.join(project_root, 'Data')
training_data_path = os.path.join(data_path,'SIIM_TRAIN_TEST')
image_dir = os.path.join(training_data_path, 'dicom-images-train')
processed_csv = os.path.join(training_data_path, 'train-rle_processed.csv')
original_csv = os.path.join(training_data_path, 'train-rle.csv')
processed_parquet = os.path.join(training_data_path, 'train-rle_processed.parquet')
