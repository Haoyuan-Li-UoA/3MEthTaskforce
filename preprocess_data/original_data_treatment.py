from data_selector import data_combination, data_path_researcher
from data_aggregator import transaction_filter, token_recording_filter, token_general_info_filter, global_data_aggregate
from feature_selector import transform_save_data, transaction_and_token_price_aggregate, transaction_and_token_general_info_aggregate, transaction_and_global_info_aggregate, generate_price_prediction_data, common_textual_info_aggregate
import warnings
import pandas as pd

# ignore DtypeWarning
warnings.filterwarnings('ignore')
LUNA_ = ["0xbd31ea8212119f94a611fa969881cba3ea06fa3d"]
LUNA = ["0xa693b19d2931d498c5b318df961919bb4aee87a5", "0xbd31ea8212119f94a611fa969881cba3ea06fa3d",
        "0xb8c77482e45f1f44de1745f52c74426c631bdd52", "0x9ff58f4ffb29fa2266ab25e75e2a8b3503311656",
        "0xae7ab96520de3a18e5e111b5eaab095312d7fe84"]

def original_data_treatment(token_num=200,
                            sparse=False,
                            random_sample=False,
                            dense=True,
                            token_list=None,
                            strategy="time_chunk_sample",
                            from_time='',
                            to_time='',
                            task='link',
                            only_consider_buy=True):

    if task == 'link':
        # pure transaction
        print("Start token transaction sampling")
        paths = data_path_researcher()
        transaction_path = paths["token_transaction_path"]
        if token_list == 'test_sample':
            transaction_df = pd.read_csv(paths["test_sample"])
            sampled_tokens = transaction_df['token_address'].unique().tolist()
        else:
            sampled_tokens = data_combination(num=token_num, sparse=sparse, random_sample=random_sample, dense=dense,
                                              token_list=token_list, path=transaction_path)
            transaction_df = transaction_filter(sampled_tokens, paths, strategy, from_time, to_time)

        # transaction + token general
        print("Statr token general information aggregation")
        general_info_path = paths["token_info_general_path"]
        general_info_dict = token_general_info_filter(sampled_tokens, general_info_path)
        transaction_df = transaction_and_token_general_info_aggregate(transaction_df, general_info_dict)
        transform_save_data(transaction_df, feature_combination="transaction", only_consider_buy=only_consider_buy)

        # transaction + token general + recording
        print('Start token price, cap and volume aggregation')
        recording_path = paths["token_info_history_path"]
        recording_dict = token_recording_filter(sampled_tokens, recording_path)
        transaction_token_recording = transaction_and_token_price_aggregate(transaction_df.copy(), recording_dict)
        transform_save_data(transaction_token_recording.copy(), feature_combination="transaction_token_recording", only_consider_buy=only_consider_buy)

        # transaction + token general + global
        print("Start token global index aggregation")
        global_data_path = paths["global_data_path"]
        global_data_df = global_data_aggregate(global_data_path)
        transaction_global = transaction_and_global_info_aggregate(transaction_df.copy(), global_data_df)
        transform_save_data(transaction_global.copy(), feature_combination="transaction_global", only_consider_buy=only_consider_buy)

        # transaction + token general + textual
        print("Statr token textual index aggregation")
        reddit_posts_sentiment_formal = paths["reddit_posts_sentiment_formal"]
        transaction_textual = common_textual_info_aggregate(transaction_df.copy(), reddit_posts_sentiment_formal)
        transform_save_data(transaction_textual.copy(), feature_combination="transaction_textual", only_consider_buy=only_consider_buy)

        # transaction + token general + recording + global
        print("Combination token recording data with global index")
        transaction_token_global_recording = transaction_and_global_info_aggregate(transaction_token_recording.copy(),
                                                                                   global_data_df)
        transform_save_data(transaction_token_global_recording.copy(),
                            feature_combination="transaction_token_global_recording", only_consider_buy=only_consider_buy)

        # transaction + token general + recording + global + textual
        print("Combination all data")
        transaction_token_all = common_textual_info_aggregate(transaction_token_global_recording.copy(),
                                                              reddit_posts_sentiment_formal)
        transform_save_data(transaction_token_all.copy(), feature_combination="transaction_token_all", only_consider_buy=only_consider_buy)

    elif task == 'price_prediction':
        # pure transaction
        print("Start token transaction sampling")
        paths = data_path_researcher()
        transaction_path = paths["token_transaction_path"]
        if token_list == 'test_sample':
            transaction_df = pd.read_csv(paths["test_sample"])
            sampled_tokens = transaction_df['token_address'].unique().tolist()
        else:
            sampled_tokens = data_combination(num=token_num, sparse=sparse, random_sample=random_sample, dense=dense,
                                              token_list=token_list, path=transaction_path)
            transaction_df = transaction_filter(sampled_tokens, paths, strategy, from_time, to_time)

        # transaction + token general
        print("Statr token general information aggregation")
        general_info_path = paths["token_info_general_path"]
        general_info_dict = token_general_info_filter(sampled_tokens, general_info_path)
        transaction_df = transaction_and_token_general_info_aggregate(transaction_df, general_info_dict)

        # transaction + token general + recording
        print('Start token price, cap and volume aggregation')
        recording_path = paths["token_info_history_path"]
        recording_dict = token_recording_filter(sampled_tokens, recording_path)
        transaction_token_recording = transaction_and_token_price_aggregate(transaction_df.copy(), recording_dict)

        # transaction + token general + global
        print("Start token global index aggregation")
        global_data_path = paths["global_data_path"]
        global_data_df = global_data_aggregate(global_data_path)

        # transaction + token general + textual
        print("Statr token textual index aggregation")
        reddit_posts_sentiment_formal = paths["reddit_posts_sentiment_formal"]

        # transaction + token general + recording + global
        print("Combination token recording data with global index")
        transaction_token_global_recording = transaction_and_global_info_aggregate(transaction_token_recording.copy(),
                                                                                   global_data_df)

        # transaction + token general + recording + global + textual
        print("Combination all data")
        transaction_token_all = common_textual_info_aggregate(transaction_token_global_recording.copy(),
                                                              reddit_posts_sentiment_formal)

        print("Generate Price Prediction Data")
        generate_price_prediction_data(transaction_token_recording.copy(), "price_prediction_transaction_token_recording", only_consider_buy)
        print("price_prediction_transaction_token_recording completed")
        generate_price_prediction_data(transaction_token_global_recording.copy(), "price_prediction_transaction_token_global_recording", only_consider_buy)
        print("price_prediction_transaction_token_global_recording completed")
        generate_price_prediction_data(transaction_token_all.copy(), "price_prediction_transaction_token_all", only_consider_buy)
        print("price_prediction_transaction_token_global_recording completed")







