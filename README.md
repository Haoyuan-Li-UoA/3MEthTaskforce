### Data Processing 

**active user address sample**
```{bash}
cd preprocess_data/
python data_sampler.py --token_list test_sample --task link_and_price_prediction --only_consider_buy
python preprocess_all_data.py
```


### Dataset Name

```python 
LINKPREDICTION_DATASETS_OPTIONS = ['crypto', 'transaction', 'transaction_token_recording', 'transaction_global', 
            'transaction_textual', 'transaction_token_global_recording', 'transaction_token_all']

PRICE_PREDICTION_OPTIONS = ['transaction_token_recording',
            'transaction_token_global_recording', 'transaction_token_all']

Model_OPTIONS = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']

Strategy_OPTIONS = ['time_chunk_sample', 'entire_token_recording']

TASK_OPTIONS = ['link', 'link_and_price_prediction']

TokenList_OPTIONS = ['test_sample', 'LUNA']
```

### Link Prediction Task Train

```{bash}
python train_link_prediction.py --dataset_name transaction_token_all --model_name DyGFormer --load_test_configs --gpu 0
```

**Downstream Task Token node Price Prediction**

```{bash}
python train_node_classification.py --dataset_name transaction_token_all --model_name DyGFormer --load_test_configs --gpu 0
```

```{bash}
cd preprocess_data/
python data_sampler.py --sparse --strategy entire_token_recording --task link_and_price_prediction --only_consider_buy
python preprocess_all_data.py
```

### Link Prediction Task Evaluation

```{bash}
python evaluate_link_prediction.py --dataset_name price_prediction_transaction_token_all --model_name DyGFormer --load_test_configs --gpu 0
```

**Sub-task: Transaction Type Prediction Evaluation**

```{bash}
python evaluate_node_classification.py --dataset_name transaction --model_name DyGFormer --load_test_configs --gpu 0
```

**LUNA EVENT Test**
```{bash}
cd preprocess_data/
python data_sampler.py --dense --strategy time_chunk_sample --task link_and_price_prediction --from_time 2022-05-05 --to_time 2022-05-11
python preprocess_all_data.py
```
