import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict

current_path = os.getcwd()

base_path = os.path.abspath(os.path.join(current_path, ".."))
folder_path = os.path.join(base_path, "3MEthTaskforce Data", "Token Info", "Token Historical Data 3880 with return")

# Define time periods
start_period2 = '2022-05-05'
end_period2 = '2022-05-11'
start_period1 = '2022-04-28'
end_period1 = '2022-05-04'

# Initialize a list to store filenames
matching_files = []

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # Assume all files are in CSV format
        file_path = os.path.join(folder_path, file_name)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure the file contains a timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Check if both time periods are present in the timestamp column
            mask_period1 = (df['timestamp'] >= start_period1) & (df['timestamp'] <= end_period1)
            mask_period2 = (df['timestamp'] >= start_period2) & (df['timestamp'] <= end_period2)

            if mask_period1.any() and mask_period2.any():
                matching_files.append(file_name)

# Define global variables to store results
luna_risk = []
common_risk = []
result = {i: [] for i in range(1, 260)}  # Weeks from 1 to 260
result_luna = {i: [] for i in range(1, 260)}  # Weeks from 1 to 260
result_common = {i: [] for i in range(1, 260)}  # Weeks from 1 to 260

# Folder path
folder_path_token_info = os.path.join(base_path, "3MEthTaskforce Data", "Token Info", "Token Historical Data 3880 with return")
current_path = os.getcwd()
base_path = os.path.abspath(os.path.join(current_path, ".."))
TY60MCD_data_path = os.path.join(base_path, "3MEthTaskforce Data", "Globa Data", "TY60MCD.csv")
file_ty60mcd = TY60MCD_data_path
# Read the TY60MCD.csv file
ty60mcd_df = pd.read_csv(file_ty60mcd)
ty60mcd_df['DateTime'] = pd.to_datetime(ty60mcd_df['DateTime'])

# Read the TY60MCD.csv file again
ty60mcd_df = pd.read_csv(file_ty60mcd)
ty60mcd_df['DateTime'] = pd.to_datetime(ty60mcd_df['DateTime'])


def calculate_return_std(prices):
    # Calculate the percentage change of prices (returns)
    returns = prices.pct_change().dropna()

    # Check if there are enough return values to calculate variance
    if len(returns) < 2:
        # If there's not enough data, return NaN or 0, depending on needs
        return 0  # Or return 0

    # Calculate variance
    variance = np.var(returns)

    return variance

# Function to remove timezone info from timestamps. All timestamps must be timezone-free.
def remove_timezone(dt):
    return dt.dt.tz_localize(None) if pd.api.types.is_datetime64tz_dtype(dt) else dt

# Function to ensure timestamps are in datetime format. All timestamps must be ensured as datetime.
def ensure_datetime(timestamp):
    return pd.to_datetime(timestamp).tz_localize(None)


def is_time_in_range(end_time, from_time=1651420800, to_time=1651939200):
    """
    Check if end_time is between from_time and to_time.

    :param from_time: Start time, datetime object
    :param to_time: End time, datetime object
    :param end_time: The time to check, datetime object
    :return: True if end_time is in the range, otherwise False
    """
    from_time = ensure_datetime(from_time)
    to_time = ensure_datetime(to_time)
    end_time = ensure_datetime(end_time)

    return from_time <= end_time <= to_time


# Function to process trade pairs and calculate risk
def process_trade_pair(start_time, end_time, price_df, ty60mcd_df):
    price_df['timestamp'] = price_df['timestamp'].dt.tz_localize(None)
    ty60mcd_df['DateTime'] = ty60mcd_df['DateTime'].dt.tz_localize(None)

    # Check time range
    ty60mcd_max_date = ensure_datetime(ty60mcd_df['DateTime'].max() + pd.DateOffset(months=1))
    price_df_max_date = ensure_datetime(price_df['timestamp'].max())
    ty60mcd_min_date = ensure_datetime(ty60mcd_df['DateTime'].min())
    price_df_min_date = ensure_datetime(price_df['timestamp'].min())

    if not (price_df_min_date <= start_time <= price_df_max_date) or \
            not (price_df_min_date <= end_time <= price_df_max_date) or \
            not (ty60mcd_min_date <= start_time <= ty60mcd_max_date) or \
            not (ty60mcd_min_date <= end_time <= ty60mcd_max_date):
        return

    weeks_diff = (end_time - start_time).days // 7 + 1
    x = min(weeks_diff, 52)

    price_period = price_df[(price_df['timestamp'] >= start_time) & (price_df['timestamp'] <= end_time)]

    if price_period.empty:
        return

    ERm = (price_period['price'].iloc[-1] - price_period['price'].iloc[0]) / price_period['price'].iloc[0]
    o2 = calculate_return_std(price_period['price'])
    if o2 == 0:
        return

    month_start = start_time.to_period('M')
    month_end = end_time.to_period('M')
    ty60mcd_period = ty60mcd_df[(ty60mcd_df['DateTime'].dt.to_period('M') >= month_start) &
                                (ty60mcd_df['DateTime'].dt.to_period('M') <= month_end)]
    Rf = float(ty60mcd_period['TY60MCD'].mean()) / 100
    r = (ERm - Rf) / o2
    return r, x


# Set up sampling strategy
def sample_strategy(token_df, ty60mcd_df, luna=True):

    if luna:
        end_times = pd.date_range(start_period2, end_period2, freq='D')
    else:
        end_times = pd.date_range(start_period1, end_period1, freq='D')

    for end_time in end_times:
        if luna:
            time_range = range(2, 27)
        else:
            time_range = range(1, 26)
        for week in time_range:  # Maximum 26 weeks
            start_time = end_time - pd.DateOffset(weeks=week)
            result_pair = process_trade_pair(start_time, end_time, token_df, ty60mcd_df)
            if result_pair is not None:  # Check if the return value is not None
                r, x = result_pair  # Unpack the result
                result[x].append(r)
                if luna:
                    luna_risk.append(r)
                else:
                    common_risk.append(r)


# Apply the sampling strategy to each file in matching_files
for file_name in tqdm(matching_files):
    file_path = os.path.join(folder_path, file_name)
    token_df = pd.read_csv(file_path)

    # Ensure the timestamps are in datetime format
    token_df['timestamp'] = pd.to_datetime(token_df['timestamp'])

    # Execute the sampling strategy
    sample_strategy(token_df, ty60mcd_df, luna=True)
    sample_strategy(token_df, ty60mcd_df, luna=False)

    # Print results or further processing

# Calculate the average for each key
result_ave = {k: np.mean(v) for k, v in result.items()}
luna_avg = np.mean(luna_risk) if luna_risk else None
common_avg = np.mean(common_risk) if common_risk else None

print(f"Result Averages: {result_ave}")
print(f"Luna Average Risk: {luna_avg}")
print(f"Common Average Risk: {common_avg}")
