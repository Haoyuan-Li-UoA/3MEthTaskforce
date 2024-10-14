import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# 提取文件名
def parse_filename(file_path):
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    if ext != '.csv':
        raise ValueError("File is not a .csv file")
    parts = name.split('-')
    token_level = parts[0]
    token_symbol = parts[1]
    return token_level, token_symbol

# 列出文件夹中的所有文件
def list_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError("Folder does not exist")
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 获取风险厌恶系数
def get_risk_aversion_coefficient(token_price_file, Rf_file, token_dict, period):
    df_60m = pd.read_csv(Rf_file)
    df = pd.read_csv(token_price_file)
    df['return'] = df['price'].pct_change()
    recent_returns = df['return'].iloc[-period:]
    volatility = recent_returns.std()
    Rf = df_60m['TY60MCD'].iloc[-period:].mean() / 100
    ERm = recent_returns.mean()
    R = (ERm - Rf) / (volatility ** 2)
    token_dict["R"] = R
    return token_dict

# 移除离群值
def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

# 计算并绘制风险厌恶系数的子图
def calculate_and_plot_risk_aversion(period_list, folder_path, Rf_file):
    file_list = list_files_in_folder(folder_path)
    token_rating_list = []

    # 获取每个文件的token信息
    for item in file_list:
        token_info = {}
        file = os.path.join(folder_path, item)
        token_info["level"], token_info["symbol"] = parse_filename(file)
        token_rating_list.append(token_info)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 创建2行4列的子图
    axes = axes.ravel()  # 展平子图数组，方便索引

    for idx, period in enumerate(period_list):
        # 针对不同的周期计算风险厌恶系数
        for item in range(len(token_rating_list)):
            file = os.path.join(folder_path, file_list[item])
            token_rating_list[item] = get_risk_aversion_coefficient(file, Rf_file, token_rating_list[item], period)

        # 按level分组
        level_groups = defaultdict(list)
        for item in token_rating_list:
            level_groups[item['level']].append(item['R'])

        # 计算每个level的平均值并移除离群值，取绝对值
        average_R = {}
        for level, values in level_groups.items():
            clean_values = remove_outliers(values)
            clean_values = np.abs(clean_values)  # 取绝对值
            # 修改 if 判断
            average_R[level] = np.mean(clean_values) if len(clean_values) > 0 else np.nan

        # 绘制level的柱状图，使用浅蓝色
        levels = list(average_R.keys())
        average_values = list(average_R.values())
        axes[idx].bar(levels, average_values, color='#ADD8E6', edgecolor='black')  # 使用浅蓝色
        axes[idx].set_xlabel('Level')
        axes[idx].set_ylabel(f'Average R (Period: {period})')
        axes[idx].set_title(f'Average R by Level (Period: {period})')

        # 按ABCD分组
        grouped_levels = defaultdict(list)
        for item in token_rating_list:
            prefix = item['level'][0]
            grouped_levels[prefix].append(item['R'])

        # 计算ABCD组的平均值，取绝对值
        average_R_grouped = {}
        for prefix, values in grouped_levels.items():
            clean_values = remove_outliers(values)
            clean_values = np.abs(clean_values)  # 取绝对值
            # 修改 if 判断
            average_R_grouped[prefix] = np.mean(clean_values) if len(clean_values) > 0 else np.nan

        # 绘制ABCD组的柱状图，使用浅蓝色
        groups = list(average_R_grouped.keys())
        average_values_grouped = list(average_R_grouped.values())
        axes[idx + 4].bar(groups, average_values_grouped, color='#ADD8E6', edgecolor='black')  # 使用浅蓝色
        axes[idx + 4].set_xlabel('Group (A, B, C, D)')
        axes[idx + 4].set_ylabel(f'Average R (Period: {period})')
        axes[idx + 4].set_title(f'Average R by Group (A, B, C, D) (Period: {period})')

    plt.tight_layout()  # 调整子图之间的间距
    plt.savefig("risk_aversion_plot.png")  # 保存图片
    plt.show()  # 确保图像显示


# 传入周期列表和文件地址，执行函数
# period_list = [30, 90, 180, 365]
# folder_path = './Data'
# Rf_file = 'TY60MCD.csv'
# calculate_and_plot_risk_aversion(period_list, folder_path, Rf_file)
