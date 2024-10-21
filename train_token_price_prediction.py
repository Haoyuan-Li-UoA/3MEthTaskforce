import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import json
import warnings

warnings.filterwarnings("ignore")


def create_lstm_model(input_size, hidden_size, output_size, num_layers=1):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.active = nn.LeakyReLU(negative_slope=0.01)
            self.active1 = nn.ReLU()

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.active(out)
            out = out[:, -1, :]
            out = self.fc(out)
            return self.active1(out)

    return LSTMModel(input_size, hidden_size, output_size, num_layers)


def create_gru_model(input_size, hidden_size, output_size, num_layers=1):
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.active = nn.LeakyReLU(negative_slope=0.01)
            self.active1 = nn.ReLU()

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.gru(x, h0)
            out = self.active(out)
            out = out[:, -1, :]
            out = self.fc(out)
            return self.active1(out)

    return GRUModel(input_size, hidden_size, output_size, num_layers)


def create_dataset_by_chunks(data, look_back_percent=10, forecast_horizon_percent=3):
    data = data[:, 1:]
    look_back = look_back_percent
    forecast_horizon = forecast_horizon_percent
    X, Y = [], []

    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back:i + look_back + forecast_horizon])

    return np.array(X), np.array(Y)


def load_and_preprocess_data(file_path, x_value):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp']).view('int64') / 1e9

    zero_ratio = (df == 0).mean()
    df = df.loc[:, zero_ratio <= x_value]
    df_sorted = df.sort_values(by='timestamp')

    keep_portion = x_value
    start_index = int(len(df_sorted) * keep_portion)
    df_recent = df_sorted.iloc[start_index:]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    df_recent_log = np.log(df_recent + 1)
    X_scaled = scaler_X.fit_transform(df_recent.values)
    Y_scaled = scaler_Y.fit_transform(df_recent.values)

    return X_scaled, Y_scaled, df_recent.values, df_recent_log.values


def train_model(model, X_train, Y_train, num_epochs=100, batch_size=32, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    X_train, Y_train = X_train.to(device), Y_train.to(device)

    train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, Y_train),
                                               batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, Y_train.shape[1] * Y_train.shape[2]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')


def test_model(model, X_test, Y_test, model_name, dataset, time_range):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.MSELoss()
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, Y_test.view(-1, Y_test.shape[1] * Y_test.shape[2]))
        print(f'Test Loss: {test_loss.item():.4f}')
        file_name = os.path.basename(dataset)
        save_model_file = f'saved_results/{model_name}/{file_name}/{time_range}_mse.json'
        results = {
            "Test Loss": test_loss.item(),
            "File Name": file_name,
            "Model Name": model_name,
            "Time Range": time_range
        }
        os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
        with open(save_model_file, 'w') as json_file:
            json.dump(results, json_file, indent=4)


def main(model_type='LSTM', csv_file='sample_time_series_price.csv', x=0.5, num_epochs=100):
    X_scaled, Y_scaled, origin, df_log = load_and_preprocess_data(csv_file, x)
    look_back = 10
    forecast_horizon = 3
    X, Y = create_dataset_by_chunks(df_log, look_back, forecast_horizon)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    X_test[torch.isnan(X_test)] = 0
    Y_test[torch.isnan(Y_test)] = 0
    X_test[torch.isinf(X_test)] = 0
    Y_test[torch.isinf(Y_test)] = 0

    X_train[torch.isnan(X_train)] = 0
    Y_train[torch.isnan(Y_train)] = 0
    X_train[torch.isinf(X_train)] = 0
    Y_train[torch.isinf(Y_train)] = 0

    input_size = X_train.shape[2]
    hidden_size = 128
    output_size = Y_train.shape[1] * Y_train.shape[2]
    num_layers = 3

    if model_type == 'LSTM':
        model = create_lstm_model(input_size, hidden_size, output_size, num_layers)
    else:
        model = create_gru_model(input_size, hidden_size, output_size, num_layers)

    train_model(model, X_train, Y_train, num_epochs=num_epochs)
    test_model(model, X_test, Y_test, model_type, csv_file, x)


# main(model_type='GRU', csv_file='./price prediction data/price.csv', x=0.56, num_epochs=10000)

Test = True

if Test:
    current_path = os.getcwd()

    base_path = os.path.abspath(os.path.join(current_path, ".."))
    price_all = os.path.join(base_path, "3MEthTaskforce Data", "Simple Test", "price_all.csv")
    price_global = os.path.join(base_path, "3MEthTaskforce Data", "Simple Test", "price_global.csv")
    price_textual = os.path.join(base_path, "3MEthTaskforce Data", "Simple Test", "price_textual.csv")
    price = os.path.join(base_path, "3MEthTaskforce Data", "Simple Test", "price.csv")
    dataset = [price_all, price_textual, price_global, price]

    model = ['GRU', 'LSTM']

    x = [0.8, 0.4, 0.0]

    for csv_file in dataset:
        for model_type in model:
            for x_value in x:
                main(model_type=model_type, csv_file=csv_file, x=x_value, num_epochs=5000)
