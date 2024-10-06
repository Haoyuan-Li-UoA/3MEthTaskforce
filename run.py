import subprocess

# 定义数据集和模型
datasets = [ 'transaction_token_global_recording', 'transaction_global',
            'transaction_textual']
models = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'TCL', 'GraphMixer', 'DyGFormer']

models_part = ['TCL', 'GraphMixer', 'DyGFormer']
# datasets_part = 'transaction_token_all'

# for model in models_part:
#     # 第一部分：train_link_prediction.py
#     command1 = [
#         'python', 'train_link_prediction.py',
#         '--dataset_name', datasets_part,
#         '--model_name', model,
#         '--load_test_configs',
#         '--gpu', '0'
#     ]
#     print(f"Running command: {' '.join(command1)}")
#     # 实时显示输出，像终端一样
#     result1 = subprocess.run(command1)
#
#     # 第二部分：train_node_classification.py
#     command2 = [
#         'python', 'train_node_classification.py',
#         '--dataset_name', datasets_part,
#         '--model_name', model,
#         '--load_test_configs',
#         '--gpu', '0'
#     ]
#     print(f"Running command: {' '.join(command2)}")
#     # 实时显示输出，像终端一样
#     result2 = subprocess.run(command2)

# 遍历数据集和模型的组合，运行命令1: train_link_prediction.py
for dataset in datasets:
    print(f"-------------------------------{dataset}--------------------------------------")
    for model in models:
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

        # 第二部分：train_node_classification.py
        command2 = [
            'python', 'train_node_classification.py',
            '--dataset_name', dataset,
            '--model_name', model,
            '--load_test_configs',
            '--gpu', '0'
        ]
        print(f"Running command: {' '.join(command2)}")
        # 实时显示输出，像终端一样
        result2 = subprocess.run(command2)
