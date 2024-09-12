import os
from preprocess_data import DATASETS


for name in DATASETS:
    os.system(f'python preprocess_data.py  --dataset_name {name}')