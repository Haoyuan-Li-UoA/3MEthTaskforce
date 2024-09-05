import os
from preprocess_data import DATASETS
from original_data_treatment import original_data_treatment

# 调用处理函数
original_data_treatment(token_num=800,
                        sparse=True,
                        random_sample=False,
                        dense=False,
                        token_list=None)

# 处理DATASETS中的每个name
for name in DATASETS:
    os.system(f'python preprocess_data.py  --dataset_name {name}')
