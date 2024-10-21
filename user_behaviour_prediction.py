import subprocess

# Define datasets and models

datasets = ['transaction', 'transaction_token_recording', 'transaction_global', 'transaction_textual',
            'transaction_token_global_recording', 'transaction_token_all']

models = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'TCL', 'DyGFormer']
# models = ['DyGFormer']

# Iterate through combinations of datasets and models, and run command 1: train_link_prediction.py
for dataset in datasets:
    for model in models:
        print(f"-------------------------------{dataset}--------------------------------------")
        print(f"-------------------------------{model}--------------------------------------")
        # Part 1: train_link_prediction.py
        command1 = [
            'python', 'train_link_prediction.py',
            '--dataset_name', dataset,
            '--model_name', model,
            '--load_test_configs',
            '--gpu', '0'
        ]
        print(f"Running command: {' '.join(command1)}")
        # Display output in real-time, like in a terminal
        result1 = subprocess.run(command1)
