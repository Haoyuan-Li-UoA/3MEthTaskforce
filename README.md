# 3MEthTaskforce: Multi-source Multi-level Multi-token Ethereum Data Platform


## Experiment Setup

**Python Version: python 3.8, Torch Version: 2.4, Compute Platform: CUDA 11.8**
```{bash}
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

```{bash}
pip install -r requirements.txt
```

## User Behaviour Prediction

### Data Processing 

**Active User Address Sample**
```{bash}
cd preprocess_data/
python data_sampler.py --token_list test_sample --task link --only_consider_buy
python preprocess_all_data.py
```

### Dataset Name

```python 
LINKPREDICTION_DATASETS_OPTIONS = ['crypto', 'transaction', 'transaction_token_recording', 'transaction_global', 
            'transaction_textual', 'transaction_token_global_recording', 'transaction_token_all']
Model_OPTIONS = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'TCL', 'DyGFormer']
Strategy_OPTIONS = ['time_chunk_sample', 'entire_token_recording']
TASK_OPTIONS = ['link', 'link_and_price_prediction']
TokenList_OPTIONS = ['test_sample']
```

### User Behaviour Prediction Task Train

**Run one model in one dataset**
```{bash}
python train_link_prediction.py --dataset_name transaction_token_all --model_name DyGFormer --load_test_configs --gpu 0
```

**Run all models in all datasets**
```{bash}
python user_behaviour_prediction.py
```

### User Behaviour Prediction Task Evaluation

```{bash}
python evaluate_link_prediction.py --dataset_name price_prediction_transaction_token_all --model_name DyGFormer --load_test_configs --gpu 0
```

## Price Prediction

```{bash}
python train_token_price_prediction.py
```

## User Behaviour Marking 

**Risk in Time Period and Luna Case Study**
```{bash}
python risk_in_time_period_and_luna_case_study.py
```

**Token Risk in Token Rating**
```{bash}
python token_rating_in_token_risk.py
```

