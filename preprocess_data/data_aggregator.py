import chardet
from data_selector import data_combination, data_path_researcher
import os
import pandas as pd
from tqdm import tqdm
from datetime import timedelta


# Function 1: transaction_filter
def transaction_filter(token_list, path, num=200):
    df_merge = pd.DataFrame()

    for token in tqdm(token_list):
        file_path = os.path.join(path, f"{token}.csv")
        df = pd.read_csv(file_path)
        df_filtered = df[['token_address', 'from_address', 'to_address', 'value', 'block_timestamp']]

        # 判断 df_filtered 的行数是否小于等于 num
        if len(df_filtered) <= num:
            # 如果 df_filtered 的数据少于或等于 num，直接全部合并
            df_merge = pd.concat([df_merge, df_filtered], ignore_index=True)
        else:
            # 如果数据多于 num，则进行采样后再合并
            df_sampled = df_filtered.sample(n=num, replace=False)
            df_merge = pd.concat([df_merge, df_sampled], ignore_index=True)

    return df_merge


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
            trust_score = token_row['trust_score'].values[0]

            # 转换 trust_score 的值
            if pd.isna(trust_score):
                trust_score = 0
            elif trust_score == "gray":
                trust_score = 1
            elif trust_score == "red":
                trust_score = 2
            elif trust_score == "yellow":
                trust_score = 3
            elif trust_score == "green":
                trust_score = 4

            token_general_dict[token] = [decimal, trust_score]

    return token_general_dict


def fix_global_data(global_data):
    # 确保 DateTime 列为 datetime 类型
    global_data['DateTime'] = pd.to_datetime(global_data['DateTime'])

    # 获取时间序列的最早和最近时间
    start_date = global_data['DateTime'].min()
    end_date = global_data['DateTime'].max()

    # 生成从最早到最近的日期范围
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # 创建一个包含所有日期的 DataFrame
    all_dates_df = pd.DataFrame(all_dates, columns=['DateTime'])

    # 合并原数据和完整日期范围
    merged_df = pd.merge(all_dates_df, global_data, on='DateTime', how='left')

    # 填补缺失值
    def fill_missing_values(df):
        # 对每一列进行处理
        for column in df.columns[1:]:
            # 填补缺失值
            df[column] = df[column].fillna(method='ffill')  # 前向填充
            df[column] = df[column].fillna(method='bfill')  # 后向填充

            # 处理特殊情况
            if df[column].isna().all():
                df[column] = 0

        return df

    # 填补缺失值
    filled_df = fill_missing_values(merged_df)

    return filled_df


# Function 4: global_data_aggregate
def global_data_aggregate(path):
    # 读取文件夹中的所有文件
    file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

    merged_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(file)

        # 将 DateTime 列转换为日期并保留时间为 00:00:00
        df['DateTime'] = pd.to_datetime(df['DateTime']).dt.floor('D')

        # 计算每天的平均值或用前一个值填充
        df_daily = df.groupby('DateTime').mean().ffill()

        # 合并数据
        if merged_df.empty:
            merged_df = df_daily
        else:
            merged_df = pd.merge(merged_df, df_daily, on='DateTime', how='outer')

    merged_df.reset_index(inplace=True)
    merged_df = fix_global_data(merged_df)
    return merged_df

def clean_data(df):
    # Step 1: 处理timestamp列，转换为datetime类型，非日期格式的行将设置为NaT
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Step 2: 移除无法转换为时间格式的行（即NaT值）
    df = df.dropna(subset=['timestamp'])

    # Step 3: 检查其余的列是否为数字
    # 遍历每一列，确保 'score', 'number_of_comment', 'positive', 'negative' 为数字
    numeric_columns = ['score', 'number_of_comment', 'positive', 'negative']

    for col in numeric_columns:
        # 将非数字的值转换为NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 4: 移除任何包含NaN的行，确保所有列都有有效值
    df = df.dropna(subset=numeric_columns)

    return df


def check_invalid_timestamps(df):
    # 将timestamp列转换为datetime格式
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # 按照timestamp排序
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 确保按天为单位的timestamp，即去除时间部分只保留日期
    df['timestamp'] = df['timestamp'].dt.floor('D')

    # Step 1: 处理重复的日期，通过对同一天的特征进行加和
    df = df.groupby('timestamp', as_index=False).sum()

    # Step 2: 生成完整的日期范围
    full_date_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')

    # Step 3: 使用完整的日期范围进行重建数据集，缺失的天数用0填充特征值
    df = df.set_index('timestamp').reindex(full_date_range).fillna(0).reset_index()
    df.rename(columns={'index': 'timestamp'}, inplace=True)

    return df



def textual_data_formula(path, k=0.5):
    # 读取CSV文件
    df = pd.read_csv(path, encoding='latin1')

    df = clean_data(df)

    # 新的DataFrame来存储处理后的数据
    processed_data = []

    # 遍历每一行，处理每个post的生命周期
    for i in range(len(df)):
        row = df.iloc[i]
        score, timestamp, number_of_comment, positive, negative = float(row['score']), row['timestamp'], float(row[
            'number_of_comment']), float(row['positive']), float(row['negative'])

        # 前三天的特征值保持不变
        for j in range(3):
            new_timestamp = timestamp + timedelta(days=j)
            processed_data.append([score, new_timestamp, number_of_comment, positive, negative])

        # 后四天的特征值按系数k递减
        for j in range(3, 7):
            score *= k
            number_of_comment *= k
            positive *= k
            negative *= k
            new_timestamp = timestamp + timedelta(days=j)
            processed_data.append([score, new_timestamp, number_of_comment, positive, negative])

        # 如果下一个post的开始时间晚于当前post的第7天之后，需要插入中间的0填充
        if i + 1 < len(df):
            next_post_timestamp = df.iloc[i + 1]['timestamp']
            last_post_day = timestamp + timedelta(days=7)
            while last_post_day < next_post_timestamp:
                processed_data.append([0, last_post_day, 0, 0, 0])
                last_post_day += timedelta(days=1)

    # 创建处理后的DataFrame
    processed_df = pd.DataFrame(processed_data,
                                columns=['score', 'timestamp', 'number_of_comment', 'positive', 'negative'])

    processed_df = check_invalid_timestamps(processed_df)

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
