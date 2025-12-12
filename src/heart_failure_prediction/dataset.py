import os

import kagglehub

from src.heart_failure_prediction.config import RAW_DATA_DIR

path = kagglehub.dataset_download(
    handle='fedesoriano/heart-failure-prediction', force_download=True
)

filename = 'heart.csv'
src_path = os.path.join(path, filename)

print(f'Dataset downloaded to: {src_path}')

dest_path = f'{RAW_DATA_DIR}/{filename}'
print(f'Moving dataset to: {dest_path}')

os.rename(src_path, f'{RAW_DATA_DIR}/{filename}')
