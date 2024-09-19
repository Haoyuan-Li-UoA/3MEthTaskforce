### Data Processing 

```{bash}
cd preprocess_data/
python data_sampler.py --strategy time_chunk_sample --luna LUNA --from_time 2022-03-01 --to_time 2022-04-30
python preprocess_all_data.py
```



### Dataset Name

```python 
DATASETS = ['crypto', 'transaction', 'transaction_token_recording', 'transaction_global', 
            'transaction_textual', 'transaction_token_global_recording', 'transaction_token_all']

Model = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']

Strategy = ['time_chunk_sample', 'entire_token_recording']

TokenSymbol = ['Terra Classic (Wormhole)', 'TerraUSD (Wormhole)', 'Tether', 'USDC', 'BNB', 'Aave WBTC', 'Lido Staked Ether']
LUNA = ["0xbd31ea8212119f94a611fa969881cba3ea06fa3d", "0xa693b19d2931d498c5b318df961919bb4aee87a5",
        "0xb8c77482e45f1f44de1745f52c74426c631bdd52", "0x9ff58f4ffb29fa2266ab25e75e2a8b3503311656",
        "0xae7ab96520de3a18e5e111b5eaab095312d7fe84"]
LUNA_Time_Period = ['2022-05-01', '2022-05-30']
```

### Link Prediction Task Train

```{bash}
python train_link_prediction.py --dataset_name transaction_token_all --model_name DyGFormer --load_test_configs --gpu 0
```

**Sub-task: Transaction Type Prediction Train**

```{bash}
python train_node_classification.py --dataset_name crypto --model_name DyGFormer --load_test_configs --gpu 0
```

```{bash}
cd preprocess_data/
python data_sampler.py --strategy time_chunk_sample --luna LUNA --from_time 2022-05-07 --to_time 2022-05-14
python preprocess_all_data.py
```

### Link Prediction Task Evaluation

```{bash}
python evaluate_link_prediction.py --dataset_name transaction --model_name DyGFormer --negative_sample_strategy random --load_test_configs --gpu 0
```

**Sub-task: Transaction Type Prediction Evaluation**

```{bash}
python evaluate_node_classification.py --dataset_name transaction --model_name DyGFormer --load_test_configs --gpu 0
```

