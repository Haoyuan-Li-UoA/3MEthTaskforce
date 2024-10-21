from data_selector import data_combination, data_path_researcher
import os
import pandas as pd
from tqdm import tqdm
from datetime import timedelta


# ----------------------------------------------------------------------------------------------------------------------

# Function 1: transaction_filter
def transaction_filter(token_list, paths=data_path_researcher(), strategy='time_chunk_sample', from_time='', to_time=''):
    def strategy_entire_token_recording(token_list, path):
        df_merge = pd.DataFrame()

        for token in tqdm(token_list):
            file_path = os.path.join(path, f"{token}.csv")
            df = pd.read_csv(file_path)
            df_filtered = df[['token_address', 'from_address', 'to_address', 'value', 'block_timestamp']]
            df_merge = pd.concat([df_merge, df_filtered], ignore_index=True)

        return df_merge

    def strategy_token_time_chunk_sample(token_list, path, from_time, to_time):
        # Initialize an empty DataFrame to store merged data
        df_merge = pd.DataFrame()

        # Ensure the 'to' date is before July 19, 2024
        deadline = pd.to_datetime("2024-07-19")
        deadline = deadline.tz_localize('UTC')
        assert to_time <= deadline, "TimePeriodError: 'to' should be the time before 19/07/2024"

        # Loop through each token address
        for token_address in tqdm(token_list):
            # Construct file path
            file_path = os.path.join(path, f"{token_address}.csv")

            # Read CSV file
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                # Keep only the required columns
                df = df[['token_address', 'from_address', 'to_address', 'value', 'block_timestamp']]

                # Convert 'block_timestamp' to datetime format
                df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])

                # Determine the range of the time chunk
                min_time = df['block_timestamp'].min()
                max_time = df['block_timestamp'].max()

                # Sample based on different conditions
                if from_time >= min_time and to_time <= max_time:
                    df_sample = df[(df['block_timestamp'] >= from_time) & (df['block_timestamp'] <= to_time)]
                elif from_time > max_time:
                    continue  # Skip if 'from' exceeds the maximum time
                elif from_time < min_time and to_time <= max_time:
                    df_sample = df[df['block_timestamp'] <= to_time]
                elif from_time >= min_time and to_time > max_time:
                    df_sample = df[df['block_timestamp'] >= from_time]
                else:  # Skip if both 'from' and 'to' are out of the time range
                    continue

                # Merge the sampled result with the DataFrame
                df_merge = pd.concat([df_merge, df_sample], ignore_index=True)

        return df_merge

    from_time = pd.to_datetime(from_time)
    to_time = pd.to_datetime(to_time)

    # Convert timestamps with time zone info to ones without it
    from_time = from_time.tz_localize('UTC')
    to_time = to_time.tz_localize('UTC')

    # Call different processing functions based on the selected strategy
    if strategy == "entire_token_recording":
        df = strategy_entire_token_recording(token_list, paths["token_transaction_path"])
    elif strategy == "time_chunk_sample":
        df = strategy_token_time_chunk_sample(token_list, paths["token_transaction_path"], from_time, to_time)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return df


# ----------------------------------------------------------------------------------------------------------------------
# Function 2: token_recording_filter
def token_recording_filter(token_list, path):
    df_dict = {}

    for token in token_list:
        file_path = os.path.join(path, f"{token}.csv")
        df = pd.read_csv(file_path)
        df = df.drop(columns='return_day')
        df_dict[token] = df

    return df_dict


# Function 3: token_general_info_filter
def token_general_info_filter(token_list, path):

    df = pd.read_csv(path, encoding="iso-8859-1")
    token_general_dict = {}

    for token in token_list:
        token_row = df[df['eth-address'] == token]
        if not token_row.empty:
            decimal = token_row['decimal'].values[0]

            token_general_dict[token] = [decimal]

    return token_general_dict


def fix_global_data(global_data):
    # Ensure 'DateTime' column is of datetime type
    global_data['DateTime'] = pd.to_datetime(global_data['DateTime'])

    # Get the earliest and latest times in the time series
    start_date = global_data['DateTime'].min()
    end_date = global_data['DateTime'].max()

    # Generate a date range from the earliest to the latest date
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a DataFrame containing all dates
    all_dates_df = pd.DataFrame(all_dates, columns=['DateTime'])

    # Merge the original data with the complete date range
    merged_df = pd.merge(all_dates_df, global_data, on='DateTime', how='left')

    # Fill in missing values
    def fill_missing_values(df):
        # Process each column
        for column in df.columns[1:]:
            # Fill missing values
            df[column] = df[column].fillna(method='ffill')  # Forward fill
            df[column] = df[column].fillna(method='bfill')  # Backward fill

            # Handle special cases
            if df[column].isna().all():
                df[column] = 0

        return df

    # Fill missing values
    filled_df = fill_missing_values(merged_df)

    return filled_df


# Function 4: global_data_aggregate
def global_data_aggregate(path):
    # Read all files in the folder
    file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

    merged_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(file)

        # Convert 'DateTime' column to date and keep the time as 00:00:00
        df['DateTime'] = pd.to_datetime(df['DateTime']).dt.floor('D')

        # Calculate the daily average or use the previous value to fill
        df_daily = df.groupby('DateTime').mean().ffill()

        # Merge data
        if merged_df.empty:
            merged_df = df_daily
        else:
            merged_df = pd.merge(merged_df, df_daily, on='DateTime', how='outer')

    merged_df.reset_index(inplace=True)
    merged_df = fix_global_data(merged_df)
    return merged_df


def clean_data(df):
    # Step 1: Handle the 'timestamp' column, convert it to datetime type, and set non-date format rows to NaT
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Step 2: Remove rows that cannot be converted to datetime format (i.e., NaT values)
    df = df.dropna(subset=['timestamp'])

    # Step 3: Check whether the remaining columns are numeric
    # Ensure 'score', 'number_of_comment', 'positive', 'negative' are numeric
    numeric_columns = ['score', 'number_of_comment', 'positive', 'negative']

    for col in numeric_columns:
        # Convert non-numeric values to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 4: Remove any rows that contain NaN, ensuring all columns have valid values
    df = df.dropna(subset=numeric_columns)

    return df


def check_invalid_timestamps(df):
    # Convert the 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Sort by 'timestamp'
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Ensure timestamps are by day, removing time parts and keeping only dates
    df['timestamp'] = df['timestamp'].dt.floor('D')

    # Step 1: Handle duplicate dates by summing the features for the same day
    df = df.groupby('timestamp', as_index=False).sum()

    # Step 2: Generate a complete date range
    full_date_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')

    # Step 3: Rebuild the dataset using the complete date range, filling in missing days with 0 for feature values
    df = df.set_index('timestamp').reindex(full_date_range).fillna(0).reset_index()
    df.rename(columns={'index': 'timestamp'}, inplace=True)

    return df


# ----------------------------------------------------------------------------------------------------------------------
def textual_data_formula(path, k=0.5):
    # Read CSV file
    df = pd.read_csv(path, encoding='latin1')

    df = clean_data(df)

    # New DataFrame to store processed data
    processed_data = []

    # Iterate through each row and process each post's lifecycle
    for i in range(len(df)):
        row = df.iloc[i]
        score, timestamp, number_of_comment, positive, negative = float(row['score']), row['timestamp'], float(row['number_of_comment']), float(row['positive']), float(row['negative'])

        # Keep the feature values unchanged for the first three days
        for j in range(3):
            new_timestamp = timestamp + timedelta(days=j)
            processed_data.append([score, new_timestamp, number_of_comment, positive, negative])

        # Decrease the feature values by a factor of k for the next four days
        for j in range(3, 7):
            score *= k
            number_of_comment *= k
            positive *= k
            negative *= k
            new_timestamp = timestamp + timedelta(days=j)
            processed_data.append([score, new_timestamp, number_of_comment, positive, negative])

        # If the start time of the next post is after the 7th day of the current post, insert 0-filled values in between
        if i + 1 < len(df):
            next_post_timestamp = df.iloc[i + 1]['timestamp']
            last_post_day = timestamp + timedelta(days=7)
            while last_post_day < next_post_timestamp:
                processed_data.append([0, last_post_day, 0, 0, 0])
                last_post_day += timedelta(days=1)

    # Create processed DataFrame
    processed_df = pd.DataFrame(processed_data,
                                columns=['score', 'timestamp', 'number_of_comment', 'positive', 'negative'])

    processed_df = check_invalid_timestamps(processed_df)

    return processed_df
# ----------------------------------------------------------------------------------------------------------------------


def common_sentiment_textual_data_formula(path, k=0.5):
    def clean_data(df):
        # Step 1: Handle the 'timestamp' column, convert it to datetime type, and set non-date format rows to NaT
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Step 2: Remove rows that cannot be converted to datetime format (i.e., NaT values)
        df = df.dropna(subset=['timestamp'])

        # Step 3: Check whether the remaining columns are numeric
        numeric_columns = ['score', 'number_of_comment', 'sentiment']

        for col in numeric_columns:
            # Convert non-numeric values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Step 4: Remove any rows that contain NaN, ensuring all columns have valid values
        df = df.dropna(subset=numeric_columns)

        return df

    def check_invalid_timestamps(df):
        # Convert the 'timestamp' column to datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Sort by 'timestamp'
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # Ensure timestamps are by day, removing time parts and keeping only dates
        df['timestamp'] = df['timestamp'].dt.floor('D')

        # Step 1: Handle duplicate dates by summing the features for the same day
        df = df.groupby('timestamp', as_index=False).sum()

        # Step 2: Generate a complete date range
        full_date_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')

        # Step 3: Rebuild the dataset using the complete date range, filling in missing days with 0 for feature values
        df = df.set_index('timestamp').reindex(full_date_range).fillna(0).reset_index()
        df.rename(columns={'index': 'timestamp'}, inplace=True)

        return df

    # Read CSV file
    df = pd.read_csv(path)

    df = clean_data(df)

    # New DataFrame to store processed data
    processed_data = []

    # Iterate through each row and process each post's lifecycle
    for i in range(len(df)):
        row = df.iloc(i)
        score, timestamp, number_of_comment, sentiment = float(row['score']), row['timestamp'], float(row['number_of_comment']), float(row['sentiment'])

        # Keep the feature values unchanged for the first three days
        for j in range(3):
            new_timestamp = timestamp + timedelta(days=j)
            processed_data.append([score, new_timestamp, number_of_comment, sentiment, 1])

        # Decrease the feature values by a factor of k for the next four days
        for j in range(3, 7):
            score *= k
            number_of_comment *= k
            new_timestamp = timestamp + timedelta(days=j)
            processed_data.append([score, new_timestamp, number_of_comment, sentiment, 1])

        # If the start time of the next post is after the 7th day of the current post, insert 0-filled values in between
        if i + 1 < len(df):
            next_post_timestamp = df.iloc[i + 1]['timestamp']
            last_post_day = timestamp + timedelta(days=7)
            while last_post_day < next_post_timestamp:
                processed_data.append([0, last_post_day, 0, 0, 1])
                last_post_day += timedelta(days=1)

    # Convert processed data into DataFrame
    processed_df = pd.DataFrame(processed_data, columns=['score', 'timestamp', 'number_of_comment', 'sentiment', 'count'])

    # Group rows with the same timestamp and take the average of specific columns ('sentiment')
    processed_df = processed_df.groupby('timestamp').agg(
        {'score': 'sum', 'number_of_comment': 'sum', 'sentiment': lambda x: x.sum() / x.count(), 'count': 'sum'}
    ).reset_index()

    # Check for invalid timestamps (customizable function)
    processed_df = check_invalid_timestamps(processed_df)

    # Compute the average sentiment
    processed_df['sentiment'] = processed_df['sentiment'] / processed_df['count']

    # Remove the 'count' column
    processed_df = processed_df.drop(columns=['count'])

    return processed_df


TEST = False

if TEST:
    paths = data_path_researcher()
    transaction_path = paths["token_transaction_path"]
    sampled_tokens = data_combination(num=3880, random_sample=True, path=transaction_path)
    # Testing the functions
    TEST1 = False

    if TEST1:
        # Test transaction_filter
        transaction_path = paths["token_transaction_path"]
        transaction_df = transaction_filter(sampled_tokens, transaction_path)
        print("Transaction Dataframe:", transaction_df.head())

    TEST2 = False

    if TEST2:
        # Test token_recording_filter
        recording_path = paths["token_info_history_path"]
        recording_dict = token_recording_filter(sampled_tokens, recording_path)
        print("Token Recording Dict:", recording_dict)

    TEST3 = False

    if TEST3:

        # Test token_general_info_filter
        general_info_path = paths["token_info_general_path"]
        general_info_dict = token_general_info_filter(sampled_tokens, general_info_path)
        print("Token General Info Dict:", general_info_dict)

    TEST4 = False

    if TEST4:

        # Test global_data_aggregate
        global_data_path = paths["global_data_path"]
        global_data_df = global_data_aggregate(global_data_path)
        print("Global Data Dataframe:", global_data_df)

    TEST5 = False

    if TEST5:

        textual_data_save = paths["textual_save_path"]
        df = textual_data_formula(textual_data_save, 0.5)
        # invalid_timestamp_rows = check_invalid_timestamps(textual_data_save)
        df.to_csv(paths["textual_formula_path"], index=False)

    TEST6 = False

    if TEST6:

        reddit_posts_sentiment_llm = paths["reddit_posts_sentiment_llm"]
        df = common_sentiment_textual_data_formula(reddit_posts_sentiment_llm, 0.5)
        # invalid_timestamp_rows = check_invalid_timestamps(textual_data_save)
        df.to_csv(paths["reddit_posts_sentiment_formal"], index=False)

