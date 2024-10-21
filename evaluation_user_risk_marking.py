import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


# Extract the filename
def parse_filename(file_path):
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    if ext != '.csv':
        raise ValueError("File is not a .csv file")
    parts = name.split('-')
    token_level = parts[0]
    token_symbol = parts[1]
    return token_level, token_symbol


# List all files in the folder
def list_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError("Folder does not exist")
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


# Get the risk aversion coefficient
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


# Remove outliers
def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]


# Calculate and plot the risk aversion coefficient in subplots
def calculate_and_plot_risk_aversion(period_list, folder_path, Rf_file):
    file_list = list_files_in_folder(folder_path)
    token_rating_list = []

    # Get token information for each file
    for item in file_list:
        token_info = {}
        file = os.path.join(folder_path, item)
        token_info["level"], token_info["symbol"] = parse_filename(file)
        token_rating_list.append(token_info)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns of subplots
    axes = axes.ravel()  # Flatten the array of subplots for easy indexing

    for idx, period in enumerate(period_list):
        # Calculate the risk aversion coefficient for different periods
        for item in range(len(token_rating_list)):
            file = os.path.join(folder_path, file_list[item])
            token_rating_list[item] = get_risk_aversion_coefficient(file, Rf_file, token_rating_list[item], period)

        # Group by level
        level_groups = defaultdict(list)
        for item in token_rating_list:
            level_groups[item['level']].append(item['R'])

        # Calculate the average value for each level, remove outliers, and take absolute values
        average_R = {}
        for level, values in level_groups.items():
            clean_values = remove_outliers(values)
            clean_values = np.abs(clean_values)  # Take absolute values
            # Modify the if condition
            average_R[level] = np.mean(clean_values) if len(clean_values) > 0 else np.nan

        # Plot the bar chart for levels using light blue color
        levels = list(average_R.keys())
        average_values = list(average_R.values())
        axes[idx].bar(levels, average_values, color='#ADD8E6', edgecolor='black')  # Use light blue
        axes[idx].set_xlabel('Level')
        axes[idx].set_ylabel(f'Average R (Period: {period})')
        axes[idx].set_title(f'Average R by Level (Period: {period})')

        # Group by A, B, C, D prefixes
        grouped_levels = defaultdict(list)
        for item in token_rating_list:
            prefix = item['level'][0]
            grouped_levels[prefix].append(item['R'])

        # Calculate the average value for A, B, C, D groups, and take absolute values
        average_R_grouped = {}
        for prefix, values in grouped_levels.items():
            clean_values = remove_outliers(values)
            clean_values = np.abs(clean_values)  # Take absolute values
            # Modify the if condition
            average_R_grouped[prefix] = np.mean(clean_values) if len(clean_values) > 0 else np.nan

        # Plot the bar chart for A, B, C, D groups using light blue color
        groups = list(average_R_grouped.keys())
        average_values_grouped = list(average_R_grouped.values())
        axes[idx + 4].bar(groups, average_values_grouped, color='#ADD8E6', edgecolor='black')  # Use light blue
        axes[idx + 4].set_xlabel('Group (A, B, C, D)')
        axes[idx + 4].set_ylabel(f'Average R (Period: {period})')
        axes[idx + 4].set_title(f'Average R by Group (A, B, C, D) (Period: {period})')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig("./Figure/risk_aversion_plot.png")  # Save the plot
    plt.show()  # Ensure the plot is displayed


period_list = [30, 90, 180, 365]

# Get current file path
current_path = os.getcwd()

# Go two directories up
base_path = os.path.abspath(os.path.join(current_path, ".."))

Rf_file = os.path.join(base_path, "3MEthTaskforce Data", "Globa Data", 'TY60MCD.csv')
folder_path = os.path.join(base_path, "3MEthTaskforce Data", "Token Info", 'Rating Data')
# folder_path = './Data'
calculate_and_plot_risk_aversion(period_list, folder_path, Rf_file)
