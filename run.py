import subprocess

# 定义数据集和模型
datasets = ['transaction_textual', 'transaction_token_all']

models = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'TCL', 'DyGFormer']

# 遍历数据集和模型的组合，运行命令1: train_link_prediction.py
for dataset in datasets:
    for model in models:
        print(f"-------------------------------{dataset}--------------------------------------")
        print(f"-------------------------------{model}--------------------------------------")
        # 第一部分：train_link_prediction.py
        command1 = [
            'python', 'train_link_prediction.py',
            '--dataset_name', dataset,
            '--model_name', model,
            '--load_test_configs',
            '--gpu', '0'
        ]
        print(f"Running command: {' '.join(command1)}")
        # 实时显示输出，像终端一样
        result1 = subprocess.run(command1)

