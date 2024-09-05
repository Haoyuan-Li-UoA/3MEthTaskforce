import os
from preprocess_data import DATASETS
from original_data_treatment import original_data_treatment
import argparse


parser = argparse.ArgumentParser('Interface for preprocessing all datasets')

parser.add_argument('--token_num', type=int, choices=DATASETS, help='Dataset Token Num', default=1000)

# 使用 store_true 来处理布尔选项
parser.add_argument('--sparse', action='store_true', help='sample sparse dataset (default: True)')
parser.add_argument('--random_sample', action='store_true', help='randomly sample dataset (default: False)')
parser.add_argument('--dense', action='store_true', help='sample dense dataset (default: False)')
parser.add_argument('--token_list', type=list, default=None, help='sample specific tokens')

args = parser.parse_args()

# 统计哪些参数被激活
conditions = [
    args.sparse,
    args.random_sample,
    args.dense,
    args.token_list is not None
]

# 确保只有一个条件为True或非None
assert sum(conditions) == 1, "You must provide only one positive value exactly in one of --sparse, --random_sample, --dense, or --token_list"


original_data_treatment(args.token_num, args.sparse, args.random_sample, args.dense, )

for name in DATASETS:
    os.system(f'python preprocess_data.py  --dataset_name {name}')
