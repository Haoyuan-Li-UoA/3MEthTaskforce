### Data Processing 

```{bash}
cd preprocess_data/
python preprocess_all_data.py
```

### Dataset Name

```python 
DATASETS = ['crypto', 'transaction', 'transaction_token_recording', 'transaction_token_general',
            'transaction_global', 'transaction_token_general_recording', 'transaction_token_general_global',
            'transaction_token_global_recording', 'transaction_token_all']
Model = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']
```

### Link Prediction Task Train

```{bash}
python train_link_prediction.py --dataset_name transaction_token_recording --model_name DyGFormer --load_test_configs --num_runs 5 --gpu 0
```

**Sub-task: Transaction Type Prediction Train**

```{bash}
python train_node_classification.py --dataset_name crypto --model_name DyGFormer --load_test_configs --num_runs 5 --gpu 0
```

### Link Prediction Task Evaluation

```{bash}
python evaluate_link_prediction.py --dataset_name crypto --model_name DyGFormer --negative_sample_strategy random --load_test_configs --num_runs 5 --gpu 0
```

**Sub-task: Transaction Type Prediction Evaluation**

```{bash}
python evaluate_node_classification.py --dataset_name crypto --model_name DyGFormer --load_test_configs --num_runs 5 --gpu 0
```

