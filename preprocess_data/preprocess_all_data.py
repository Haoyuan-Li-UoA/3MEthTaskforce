import os
from preprocess_data import DATASETS
from original_data_treatment import original_data_treatment

original_data_treatment()

for name in DATASETS:
    os.system(f'python preprocess_data.py  --dataset_name {name}')
