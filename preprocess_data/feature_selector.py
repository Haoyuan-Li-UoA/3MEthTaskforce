import pandas as pd
from data_selector import data_combination, data_path_researcher
from data_aggregator import transaction_filter, token_recording_filter, token_general_info_filter, global_data_aggregate
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os


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
    # 确保decimal列存在并且是整数类型
    if 'decimal' in df.columns:
        # 将decimal列转换为浮点数，避免整数负指数问题
        df['decimal'] = df['decimal'].astype(float)

        # 计算10的负decimal次方
        df['adjustment'] = 10.0 ** (-df['decimal'])

        # 确保value列为数值类型，非数值类型将转换为NaN
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # 计算新的value值
        df['value'] = df['value'] * df['adjustment']

        # 删除decimal列和调整列
        df = df.drop(columns=['decimal', 'adjustment'])

    return df


# Function 2: transaction_and_token_general_info_aggregate
def transaction_and_token_general_info_aggregate(transaction, token_general_info):

    transaction_general = transaction
    decimal = []
    trust_score = []

    # 遍历transaction的每一条记录
    for _, row in tqdm(transaction_general.iterrows()):
        token_address = row['token_address']

        # 获取对应的token general info数据
        if token_address in token_general_info:
            decimal.append(token_general_info[token_address][0])
            trust_score.append(token_general_info[token_address][1])
        else:
            decimal.append(0)
            trust_score.append(0)

    transaction_general['decimal'] = decimal
    transaction_general['trust_score'] = trust_score

    transaction_general = adjust_values(transaction_general)

    return transaction_general


# Function 3: transaction_and_global_info_aggregate
def transaction_and_global_info_aggregate(transaction, global_info):
    # 确保时间戳格式正确

    transaction_global = transaction

    transaction_global['block_timestamp'] = pd.to_datetime(transaction_global['block_timestamp'])
    global_info['DateTime'] = pd.to_datetime(global_info['DateTime'])

    # 初始化新列
    global_columns = global_info.columns.difference(['DateTime'])
    for col in global_columns:
        transaction_global[col] = None

    # 按日期匹配global info数据
    for idx, row in tqdm(transaction_global.iterrows()):
        trans_date = row['block_timestamp'].date()

        # 找到对应日期的global info
        global_record = global_info[global_info['DateTime'].dt.date == trans_date]

        if not global_record.empty:
            for col in global_columns:
                transaction_global.at[idx, col] = global_record[col].values[0]

    return transaction_global


def transaction_and_textual_info_aggregate(transaction, path):
    # 读取textual数据
    textual_df = pd.read_csv(path)

    textual_df = textual_df.fillna(0)

    # 确保timestamp和block_timestamp转换为日期格式
    transaction['block_timestamp'] = pd.to_datetime(transaction['block_timestamp']).dt.date
    textual_df['timestamp'] = pd.to_datetime(textual_df['timestamp']).dt.date

    # 准备空列表来存储textual数据对应的特征
    score_list = []
    comment_list = []
    positive_list = []
    negative_list = []

    # 遍历transaction的每一条记录
    for _, row in tqdm(transaction.iterrows()):
        transaction_date = row['block_timestamp']

        # 获取对应的textual信息（按天）
        textual_info = textual_df[textual_df['timestamp'] == transaction_date]

        if not textual_info.empty:
            # 提取对应日期的特征
            score_list.append(textual_info['score'].values[0])
            comment_list.append(textual_info['number_of_comment'].values[0])
            positive_list.append(textual_info['positive'].values[0])
            negative_list.append(textual_info['negative'].values[0])
        else:
            # 如果没有对应日期的textual信息，则填充None或其他默认值
            score_list.append(0)
            comment_list.append(0)
            positive_list.append(0)
            negative_list.append(0)

    # 将textual特征添加到transaction数据中
    transaction['textual_score'] = score_list
    transaction['textual_comment'] = comment_list
    transaction['textual_positive'] = positive_list
    transaction['textual_negative'] = negative_list

    # 返回合并后的DataFrame
    return transaction


def remove_records_after_timestamp(df, timestamp="2024-07-23 00:00:00"):
    # 确保 block_timestamp 列为 datetime 类型
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])

    # 筛选出在指定时间点之后的记录
    filtered_df = df[df['block_timestamp'] <= timestamp]

    return filtered_df


def create_or_get_feature_folder(feature_combination):
    # 1. 找到上一级路径
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # 2. 进入 Origin Data 文件夹
    origin_data_path = os.path.join(parent_dir, 'Origin Data')
    if not os.path.exists(origin_data_path):
        raise FileNotFoundError(f"'Origin Data' folder not exist: {origin_data_path}")

    # 3. 检查是否存在 feature_combination 同名文件夹
    feature_folder_path = os.path.join(origin_data_path, feature_combination)

    # 4. 如果没有同名文件夹就创建一个，如果有就进入
    if not os.path.exists(feature_folder_path):
        os.makedirs(feature_folder_path)

    feature_folder_path = feature_folder_path + "./"

    # 5. 返回路径
    return feature_folder_path

def transform_save_data(data, path="./", feature_combination='test'):

    data = data.sort_values(by='block_timestamp', ascending=True)

    data = remove_records_after_timestamp(data)

    # 3. 归一化非基本列
    basic_columns = ['token_address', 'from_address', 'to_address', 'block_timestamp']
    feature_columns = data.columns.difference(basic_columns)
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    # 4. 创建新的标准数据帧
    records = []
    for _, row in data.iterrows():
        timestamp_unix = int(row['block_timestamp'].timestamp() * 1000)  # 转换为13位时间序列

        for user, state_label in [(row['from_address'], 0), (row['to_address'], 1)]:
            record = [
                         user,                    # user
                         row['token_address'],    # item
                         timestamp_unix,          # timestamp
                         state_label              # state_label
                     ] + list(row[feature_columns])
            records.append(record)

    # 创建DataFrame
    standard_data = pd.DataFrame(records, columns=['user', 'item', 'timestamp', 'state_label'] + list(feature_columns))

    # 5. 将user address 和 item address 转换为数字
    user_to_id = {user: idx for idx, user in enumerate(standard_data['user'].unique())}
    item_to_id = {item: idx for idx, item in enumerate(standard_data['item'].unique())}

    standard_data['user'] = standard_data['user'].map(user_to_id)
    standard_data['item'] = standard_data['item'].map(item_to_id)

    file_name = feature_combination + ".csv"

    path = create_or_get_feature_folder(feature_combination)

    standard_data.to_csv(path+file_name, index=False)


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




