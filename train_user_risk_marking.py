import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# 定义文件夹路径
folder_path = r'E:\Python Workspace\Python Virtual Environment\3MEthTaskforce Data\Token Info\Token Historical Data 3880 with return'

# 定义两个时间段
start_period2 = '2022-05-05'
end_period2 = '2022-05-13'
start_period1 = '2022-04-26'
end_period1 = '2022-05-04'

# 初始化一个存储文件名的列表
matching_files = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # 假设所有文件为CSV格式
        file_path = os.path.join(folder_path, file_name)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 确保文件中有timestamp列
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # 检查两个时间段是否在文件的timestamp列中
            mask_period1 = (df['timestamp'] >= start_period1) & (df['timestamp'] <= end_period1)
            mask_period2 = (df['timestamp'] >= start_period2) & (df['timestamp'] <= end_period2)

            if mask_period1.any() and mask_period2.any():
                matching_files.append(file_name)

# 输出包含两个时间段的文件列表
print("Files containing both time periods:", matching_files)


# 定义存储结果的全局变量
luna_risk = []
common_risk = []
result = {i: [] for i in range(1, 260)}  # 1 到 260 之间的周数
result_luna = {i: [] for i in range(1, 260)}  # 1 到 260 之间的周数
result_common = {i: [] for i in range(1, 260)}  # 1 到 260 之间的周数

# 文件夹路径
folder_path_token_info = r'E:\Python Workspace\Python Virtual Environment\3MEthTaskforce Data\Token Info\Token Historical Data 3880 with return'
current_path = os.getcwd()
base_path = os.path.abspath(os.path.join(current_path, ".."))
TY60MCD_data_path = os.path.join(base_path, "3MEthTaskforce Data", "Globa Data", "TY60MCD.csv")
file_ty60mcd = TY60MCD_data_path
# 读取TY60MCD.csv文件
ty60mcd_df = pd.read_csv(file_ty60mcd)
ty60mcd_df['DateTime'] = pd.to_datetime(ty60mcd_df['DateTime'])

# 读取TY60MCD.csv文件
ty60mcd_df = pd.read_csv(file_ty60mcd)
ty60mcd_df['DateTime'] = pd.to_datetime(ty60mcd_df['DateTime'])


def calculate_return_std(prices):
    # 计算价格的百分比变化（收益率）
    returns = prices.pct_change().dropna()

    # 检查是否有足够的返回值进行方差计算
    if len(returns) < 2:
        # 如果没有足够的变化率数据，返回 NaN 或者 0，视需求而定
        return 0  # 或者 return 0

    # 计算方差
    variance = np.var(returns)

    return variance

# 去除时间戳中的时区信息的函数 所有时间戳都必须没有时区信息
def remove_timezone(dt):
    return dt.dt.tz_localize(None) if pd.api.types.is_datetime64tz_dtype(dt) else dt

# 确保时间为datetime格式的函数 所有时间戳都必须确保是时间戳
def ensure_datetime(timestamp):
    return pd.to_datetime(timestamp).tz_localize(None)


def is_time_in_range(end_time, from_time=1651420800, to_time=1651939200):
    """
    判断 end_time 是否在 from_time 和 to_time 之间

    :param from_time: 起始时间，datetime对象
    :param to_time: 结束时间，datetime对象
    :param end_time: 需要判断的时间，datetime对象
    :return: 如果 end_time 在范围内，返回 True，否则返回 False
    """
    from_time = ensure_datetime(from_time)
    to_time = ensure_datetime(to_time)
    end_time = ensure_datetime(end_time)

    return from_time <= end_time <= to_time


# 处理交易对的函数，计算风险
def process_trade_pair(start_time, end_time, price_df, ty60mcd_df):
    price_df['timestamp'] = price_df['timestamp'].dt.tz_localize(None)
    ty60mcd_df['DateTime'] = ty60mcd_df['DateTime'].dt.tz_localize(None)

    # 检查时间范围
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


# 设置采样策略
def sample_strategy(token_df, ty60mcd_df, luna=True):

    if luna:
        end_times = pd.date_range('2022-05-05', '2022-05-11', freq='D')
    else:
        end_times = pd.date_range('2022-04-28', '2022-05-04', freq='D')

    for end_time in end_times:
        if luna:
            time_range = range(2, 27)
        else:
            time_range = range(1, 26)
        for week in time_range:  # 最长26周
            start_time = end_time - pd.DateOffset(weeks=week)
            # r, x = process_trade_pair(start_time, end_time, token_df, ty60mcd_df)
            result_pair = process_trade_pair(start_time, end_time, token_df, ty60mcd_df)
            if result_pair is not None:  # 检查返回值是否为None
                r, x = result_pair  # 解包结果
                # print(result_pair)
                result[x].append(r)
                if luna:
                    luna_risk.append(r)
                else:
                    common_risk.append(r)


# 对 matching_files 中的每个文件执行采样策略
for file_name in tqdm(matching_files):
    file_path = os.path.join(folder_path, file_name)
    token_df = pd.read_csv(file_path)

    # 确保时间戳为 datetime 格式
    token_df['timestamp'] = pd.to_datetime(token_df['timestamp'])

    # 执行采样策略
    sample_strategy(token_df, ty60mcd_df, luna=True)
    sample_strategy(token_df, ty60mcd_df, luna=False)

    # 打印结果或进一步处理


# 计算每个key对应的平均值
result_ave = {k: np.mean(v) for k, v in result.items()}

luna_avg = np.mean(luna_risk) if luna_risk else None
common_avg = np.mean(common_risk) if common_risk else None


print(f"Result Averages: {result_ave}")
print(f"Luna Average Risk: {luna_avg}")
print(f"Common Average Risk: {common_avg}")