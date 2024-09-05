from data_selector import data_combination, data_path_researcher
from data_aggregator import transaction_filter, token_recording_filter, token_general_info_filter, global_data_aggregate
from feature_selector import transform_save_data, transaction_and_token_price_aggregate, transaction_and_token_general_info_aggregate, transaction_and_global_info_aggregate


def original_data_treatment(token_num, sparse=True, random_sample=False, dense=False, token_list=None):

    paths = data_path_researcher()
    transaction_path = paths["token_transaction_path"]
    sampled_tokens = data_combination(num=token_num, sparse=sparse, random_sample=random_sample, dense=dense, token_list=token_list, path=transaction_path)
    transaction_df = transaction_filter(sampled_tokens, transaction_path)
    transform_save_data(transaction_df, feature_combination="transaction")


    recording_path = paths["token_info_history_path"]
    recording_dict = token_recording_filter(sampled_tokens, recording_path)
    transaction_token_recording = transaction_and_token_price_aggregate(transaction_df.copy(), recording_dict)
    transform_save_data(transaction_token_recording, feature_combination="transaction_token_recording")


    general_info_path = paths["token_info_general_path"]
    general_info_dict = token_general_info_filter(sampled_tokens, general_info_path)
    transaction_token_general = transaction_and_token_general_info_aggregate(transaction_df.copy(), general_info_dict)
    transform_save_data(transaction_token_general, feature_combination="transaction_token_general")


    global_data_path = paths["global_data_path"]
    global_data_df = global_data_aggregate(global_data_path)
    transaction_global = transaction_and_global_info_aggregate(transaction_df.copy(), global_data_df)
    transform_save_data(transaction_global, feature_combination="transaction_global")


    transaction_token_general_recording = transaction_and_token_general_info_aggregate(transaction_token_recording.copy(), general_info_dict)
    transform_save_data(transaction_token_general_recording, feature_combination="transaction_token_general_recording")


    transaction_token_general_global = transaction_and_token_general_info_aggregate(transaction_global, general_info_dict)
    transform_save_data(transaction_token_general_global, feature_combination="transaction_token_general_global")


    transaction_token_global_recording = transaction_and_global_info_aggregate(transaction_token_recording.copy(), global_data_df)
    transform_save_data(transaction_token_global_recording, feature_combination="transaction_token_global_recording")


    transaction_token_all = transaction_and_token_general_info_aggregate(transaction_token_global_recording, general_info_dict)
    transform_save_data(transaction_token_all, feature_combination="transaction_token_all")






