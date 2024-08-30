import pandas as pd
from tqdm import tqdm

# 创建一个空的 DataFrame 用于存储标准数据
# standard_data = pd.DataFrame(columns=['user', 'item', 'timestamp', 'state_label', 'features'])
def process_data_with_timestamps(df, features):
    # 创建一个存储标准数据的空列表
    standard_data_list = []

    # 遍历每一行记录并转换
    for _, row in tqdm(df.iterrows()):
        # 将时间戳转换为11413位Unix毫秒级别时间戳
        timestamp_in_ms = pd.to_datetime(row['timeStamp']).timestamp() * 1000

        # 如果需要整数表示，取消小数位
        timestamp_in_ms = int(timestamp_in_ms)

        # 创建两行标准数据
        row_from = {
            'user': row['from'],
            'item': row['contractAddress'],
            'timestamp': timestamp_in_ms,
            'state_label': 0,
            'features': 0
        }

        row_to = {
            'user': row['to'],
            'item': row['contractAddress'],
            'timestamp': timestamp_in_ms,
            'state_label': 1,
            'features': 0
        }

        # 添加到列表中
        standard_data_list.append(row_from)
        standard_data_list.append(row_to)

    # 将列表转换为 DataFrame
    standard_data = pd.DataFrame(standard_data_list)

    return standard_data


def convert_addresses_to_numbers(df):
    """
    将DataFrame中的地址字符串转换为整数表示。
    默认将第一列和第二列视为user address和item address。

    参数：
    df: pd.DataFrame - 输入的数据框，假定第一列为用户地址，第二列为item地址。

    返回：
    pd.DataFrame - 地址转换后的数据框。
    """
    # 提取用户和item列
    user_col = df.columns[0]
    item_col = df.columns[1]

    # 创建地址到整数的映射
    user_address_to_number = {address: idx for idx, address in enumerate(df[user_col].unique())}
    # user_address_to_number = {address: idx + 1 for idx, address in enumerate(df[user_col].unique())}
    item_address_to_number = {address: idx for idx, address in enumerate(df[item_col].unique())}
    # item_address_to_number = {address: idx + 1 for idx, address in enumerate(df[item_col].unique())}

    # 映射转换
    df[user_col] = df[user_col].map(user_address_to_number)
    df[item_col] = df[item_col].map(item_address_to_number)

    sorted_df = df.sort_values(by='timestamp', ascending=True)

    return sorted_df
