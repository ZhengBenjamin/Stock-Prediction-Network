import os
import subprocess
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class StockPreprocessor:
    def __init__(self, data_dir="data", test_size=0.2, sequence_length=10):
        self.data_dir = data_dir
        self.test_size = test_size
        self.sequence_length = sequence_length
        self._ensure_data()

    def _ensure_data(self):
        # Check if there are any CSV files in the data directory
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if not csv_files:
            print("No data found. Downloading")
            subprocess.run(["python", "downloadData.py"], check=True)
        else:
            print("Data found.")

    def load_stock_data(self, filename):
        df = pd.read_csv(os.path.join(self.data_dir, filename))
        # Sort by date just in case
        df = df.sort_values("Date")
        # Drop Date column, keep numerical features
        df = df.drop(columns=["Date"])
        return df.values.astype(np.float32)

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length][5])  # Predict 'Close' price
        return np.array(X), np.array(y)

    def preprocess_all(self):
        stocks = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        all_train, all_test = {}, {}
        for stock in stocks:
            print(f"Processing {stock}...")
            data = self.load_stock_data(stock)
            X, y = self.create_sequences(data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )
            all_train[stock] = (X_train, y_train)
            all_test[stock] = (X_test, y_test)
        return all_train, all_test
