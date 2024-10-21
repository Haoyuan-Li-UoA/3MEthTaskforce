import pandas as pd
from data_selector import data_combination, data_path_researcher
from data_aggregator import transaction_filter, token_recording_filter, token_general_info_filter, global_data_aggregate
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np


# Function 1: transaction_and_token_price_aggregate
def transaction_and_token_price_aggregate(transaction, token_price):

    transaction_price = transaction
    for index, row in tqdm(transaction_price.iterrows()):
        token_address = row['token_address']
        block_timestamp = pd.to_datetime(row['block_timestamp']).date()

        if token_address in token_price:
            token_price_df = token_price[token_address]
            # Ensure the 'timestamp' column is datetime type
            token_price_df['timestamp'] = pd.to_datetime(token_price_df['timestamp'])

            # Find the price record for the closest previous date
            price_record = token_price_df[token_price_df['timestamp'].dt.date == block_timestamp]

            if not price_record.empty:
                # Assuming we want the first record if multiple exist
                closest_price = price_record.iloc[0]
                transaction_price.loc[index, 'price'] = closest_price['price']
                transaction_price.loc[index, 'market_caps'] = closest_price['market_caps']
                transaction_price.loc[index, 'total_volumes'] = closest_price['total_volumes']
            else:
                # Handle case where no price is found for the specific date
                transaction_price.loc[index, 'price'] = 0
                transaction_price.loc[index, 'market_caps'] = 0
                transaction_price.loc[index, 'total_volumes'] = 0

    return transaction_price


def adjust_values(df):
    # Ensure the 'decimal' column exists and is of integer type
    if 'decimal' in df.columns:
        # Convert 'decimal' column to float to avoid negative exponent issues with integers
        df['decimal'] = df['decimal'].astype(float)

        # Calculate 10 raised to the negative 'decimal' power
        df['adjustment'] = 10.0 ** (-df['decimal'])

        # Ensure 'value' column is numeric, convert non-numeric types to NaN
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Compute new 'value' based on 'adjustment'
        df['value'] = df['value'] * df['adjustment']

        # Drop 'decimal' and 'adjustment' columns
        df = df.drop(columns=['decimal', 'adjustment'])

    return df


# Function 2: transaction_and_token_general_info_aggregate
def transaction_and_token_general_info_aggregate(transaction, token_general_info):

    transaction_general = transaction
    decimal = []
    trust_score = []

    # Iterate through each record in the transaction
    for _, row in tqdm(transaction_general.iterrows()):
        token_address = row['token_address']

        # Get corresponding token general info data
        if token_address in token_general_info:
            decimal.append(token_general_info[token_address][0])
        else:
            decimal.append(0)

    transaction_general['decimal'] = decimal

    transaction_general = adjust_values(transaction_general)

    return transaction_general


# Function 3: transaction_and_global_info_aggregate
def transaction_and_global_info_aggregate(transaction, global_info):
    # Ensure timestamp formats are correct

    transaction_global = transaction

    transaction_global['block_timestamp'] = pd.to_datetime(transaction_global['block_timestamp'])
    global_info['DateTime'] = pd.to_datetime(global_info['DateTime'])

    # Initialize new columns
    global_columns = global_info.columns.difference(['DateTime'])
    for col in global_columns:
        transaction_global[col] = None

    # Match global info data by date
    for idx, row in tqdm(transaction_global.iterrows()):
        trans_date = row['block_timestamp'].date()

        # Find the corresponding global info for the date
        global_record = global_info[global_info['DateTime'].dt.date == trans_date]

        if not global_record.empty:
            for col in global_columns:
                transaction_global.at[idx, col] = global_record[col].values[0]

    return transaction_global


def transaction_and_textual_info_aggregate(transaction, path):
    # Read textual data
    textual_df = pd.read_csv(path)

    textual_df = textual_df.fillna(0)

    # Ensure 'timestamp' and 'block_timestamp' are converted to date format
    transaction['block_timestamp'] = pd.to_datetime(transaction['block_timestamp']).dt.date
    textual_df['timestamp'] = pd.to_datetime(textual_df['timestamp']).dt.date

    # Prepare empty lists to store the corresponding textual data features
    score_list = []
    comment_list = []
    positive_list = []
    negative_list = []

    # Iterate through each record in the transaction
    for _, row in tqdm(transaction.iterrows()):
        transaction_date = row['block_timestamp']

        # Get the corresponding textual info by day
        textual_info = textual_df[textual_df['timestamp'] == transaction_date]

        if not textual_info.empty:
            # Extract the features for the corresponding date
            score_list.append(textual_info['score'].values[0])
            comment_list.append(textual_info['number_of_comment'].values[0])
            positive_list.append(textual_info['positive'].values[0])
            negative_list.append(textual_info['negative'].values[0])
        else:
            # If no textual info is found for the date, fill with None or default values
            score_list.append(0)
            comment_list.append(0)
            positive_list.append(0)
            negative_list.append(0)

    # Add textual features to the transaction data
    transaction['textual_score'] = score_list
    transaction['textual_comment'] = comment_list
    transaction['textual_positive'] = positive_list
    transaction['textual_negative'] = negative_list

    # Return the merged DataFrame
    return transaction


def common_textual_info_aggregate(transaction, path):
    # Read textual data
    textual_df = pd.read_csv(path)

    textual_df = textual_df.fillna(0)

    # Ensure 'timestamp' and 'block_timestamp' are converted to date format
    transaction['block_timestamp'] = pd.to_datetime(transaction['block_timestamp']).dt.date
    textual_df['timestamp'] = pd.to_datetime(textual_df['timestamp']).dt.date

    # Prepare empty lists to store the corresponding textual data features
    score_list = []
    comment_list = []
    sentiment_list = []

    # Iterate through each record in the transaction
    for _, row in tqdm(transaction.iterrows()):
        transaction_date = row['block_timestamp']

        # Get the corresponding textual info by day
        textual_info = textual_df[textual_df['timestamp'] == transaction_date]

        if not textual_info.empty:
            # Extract the features for the corresponding date
            score_list.append(textual_info['score'].values[0])
            comment_list.append(textual_info['number_of_comment'].values[0])
            sentiment_list.append(textual_info['sentiment'].values[0])
        else:
            # If no textual info is found for the date, fill with None or default values
            score_list.append(0)
            comment_list.append(0)
            sentiment_list.append(0)

    # Add textual features to the transaction data
    transaction['textual_score'] = score_list
    transaction['textual_comment'] = comment_list
    transaction['sentiment'] = sentiment_list

    # Return the merged DataFrame
    return transaction


def remove_records_after_timestamp(df, timestamp="2024-07-23 00:00:00"):
    # Ensure the 'block_timestamp' column is of datetime type
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])

    # Filter out records after the specified timestamp
    filtered_df = df[df['block_timestamp'] <= timestamp]

    return filtered_df


def create_or_get_feature_folder(feature_combination):
    # 1. Find the parent directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # 2. Navigate to the Origin Data folder
    origin_data_path = os.path.join(parent_dir, 'Origin Data')
    if not os.path.exists(origin_data_path):
        raise FileNotFoundError(f"'Origin Data' folder not exist: {origin_data_path}")

    # 3. Check if a folder with the same name as feature_combination exists
    feature_folder_path = os.path.join(origin_data_path, feature_combination)

    # 4. If no folder exists, create one, otherwise navigate into it
    if not os.path.exists(feature_folder_path):
        os.makedirs(feature_folder_path)

    feature_folder_path = feature_folder_path + "./"

    # 5. Return the path
    return feature_folder_path


def transform_save_data(data, feature_combination='test', only_consider_buy=True):

    data = data.sort_values(by='block_timestamp', ascending=True)

    data = remove_records_after_timestamp(data)

    # 3. Normalize non-basic columns
    basic_columns = ['token_address', 'from_address', 'to_address', 'block_timestamp']
    feature_columns = data.columns.difference(basic_columns)
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    # data[feature_columns] = np.log(data[feature_columns] + 1)

    # 4. Create a new standardized DataFrame
    records = []
    for _, row in data.iterrows():
        timestamp_unix = int(row['block_timestamp'].timestamp() * 1000)  # Convert to 13-digit timestamp
        if only_consider_buy:
            for user, state_label in [(row['to_address'], 1)]:
                record = [
                             user,  # user
                             row['token_address'],  # item
                             timestamp_unix,  # timestamp
                             state_label  # state_label
                         ] + list(row[feature_columns])
                records.append(record)
        else:
            for user, state_label in [(row['from_address'], 0), (row['to_address'], 1)]:
                record = [
                            user,                    # user
                            row['token_address'],    # item
                            timestamp_unix,          # timestamp
                            state_label              # state_label
                        ] + list(row[feature_columns])
                records.append(record)

    # Create a DataFrame
    standard_data = pd.DataFrame(records, columns=['user', 'item', 'timestamp', 'state_label'] + list(feature_columns))

    # 5. Convert user address and item address to numeric
    user_to_id = {user: idx for idx, user in enumerate(standard_data['user'].unique())}
    item_to_id = {item: idx for idx, item in enumerate(standard_data['item'].unique())}

    standard_data['user'] = standard_data['user'].map(user_to_id)
    standard_data['item'] = standard_data['item'].map(item_to_id)

    file_name = feature_combination + ".csv"

    path = create_or_get_feature_folder(feature_combination)

    standard_data.to_csv(path+file_name, index=False)

    return standard_data


def generate_price_prediction_data(data, feature_combination='price_prediction', only_consider_buy=True):

    data = data.sort_values(by='block_timestamp', ascending=True)

    data = remove_records_after_timestamp(data)

    # 3. Normalize non-basic columns
    basic_columns = ['token_address', 'from_address', 'to_address', 'block_timestamp', 'price']
    feature_columns = data.columns.difference(basic_columns)
    scaler = MinMaxScaler(feature_range=(0, 1000))
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    # 4. Create a new standardized DataFrame
    records = []
    for _, row in data.iterrows():
        timestamp_unix = int(row['block_timestamp'].timestamp() * 1000)  # Convert to 13-digit timestamp
        if only_consider_buy:
            for user, state_label in [(row['to_address'], 1)]:
                record = [
                            user,                    # user
                            row['token_address'],    # item
                            timestamp_unix,          # timestamp
                            row['price'],            # price
                            state_label              # state_label
                        ] + list(row[feature_columns])
                records.append(record)
        else:
            for user, state_label in [(row['from_address'], 0), (row['to_address'], 1)]:
                record = [
                            user,                    # user
                            row['token_address'],    # item
                            timestamp_unix,          # timestamp
                            row['price'],            # price
                            state_label              # state_label
                        ] + list(row[feature_columns])
                records.append(record)

    # Create a DataFrame
    standard_data = pd.DataFrame(records, columns=['user', 'item', 'timestamp', 'price', 'state_label'] + list(feature_columns))

    # 5. Convert user address and item address to numeric
    user_to_id = {user: idx for idx, user in enumerate(standard_data['user'].unique())}
    item_to_id = {item: idx for idx, item in enumerate(standard_data['item'].unique())}

    standard_data['user'] = standard_data['user'].map(user_to_id)
    standard_data['item'] = standard_data['item'].map(item_to_id)

    scaler = MinMaxScaler()
    standard_data[feature_columns] = scaler.fit_transform(standard_data[feature_columns])

    # Construct file name
    file_name = feature_combination + ".csv"

    # Get save path
    path = create_or_get_feature_folder(feature_combination)

    # Save data as CSV file
    standard_data.to_csv(os.path.join(path, file_name), index=False)


TEST = False

if TEST:
    paths = data_path_researcher()
    transaction_path = paths["token_transaction_path"]
    sampled_tokens = data_combination(num=200, sparse=True, path=transaction_path)
    transaction_df = transaction_filter(sampled_tokens, transaction_path, num=200)
    TEST1 = False
    # transform_save_data(transaction_df)

    if TEST1:
        recording_path = paths["token_info_history_path"]
        recording_dict = token_recording_filter(sampled_tokens, recording_path)
        df = transaction_and_token_price_aggregate(transaction_df, recording_dict)
        transform_save_data(df)
        # print(df)

    TEST2 = False

    if TEST2:
        general_info_path = paths["token_info_general_path"]
        general_info_dict = token_general_info_filter(sampled_tokens, general_info_path)
        df = transaction_and_token_general_info_aggregate(transaction_df, general_info_dict)
        transform_save_data(df)
        # print(df)

    TEST3 = False

    if TEST3:
        global_data_path = paths["global_data_path"]
        global_data_df = global_data_aggregate(global_data_path)
        df = transaction_and_global_info_aggregate(transaction_df, global_data_df)
        transform_save_data(df)
        # print(df)

    TEST4 = False

    if TEST4:
        textual_formula_path = paths["textual_formula_path"]
        df = transaction_and_textual_info_aggregate(transaction_df, textual_formula_path)
        transform_save_data(df)

