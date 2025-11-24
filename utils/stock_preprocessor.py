import os
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from .download_data import DownloadData
from typing import List, Optional, LiteralString

class StockPreprocessor:
    def __init__(self, data_dir: Optional[str] = "data", sequence_length: Optional[int] = 1000):
        self.downloader = DownloadData()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.normalized_data = None
        self._ensure_data()

    def get_normalized_data(self) -> np.ndarray:
        if self.normalized_data is not None:
            return self.normalized_data
        
        data = self.parse_data()
        normalized_data = self.normalize(data)
        self.normalized_data = normalized_data
        return normalized_data

    def parse_data(self) -> np.ndarray:
        stock_data = []

        for file in os.listdir(self.data_dir):
            data = self.get_data_arr(file)

            if data is not None:
                stock_data.append(data)
        
        return np.array(stock_data)
    
    def get_close_val(self, stock_idx: int, time_idx: int) -> float:
        if self.normalized_data is None:
            self.get_normalized_data()
        
        return self.normalized_data[stock_idx, time_idx, 4]
    
    def normalize(self, stock_data: np.ndarray) -> np.ndarray:
        normalized_data = np.zeros_like(stock_data)
        num_features = stock_data.shape[2]

        for feature_idx in range(num_features):
            feature_vals = stock_data[:, :, feature_idx].reshape(-1, 1)
            scaler = MinMaxScaler()
            normalized_feature = scaler.fit_transform(feature_vals).reshape(stock_data.shape[0], stock_data.shape[1])
            normalized_data[:, :, feature_idx] = normalized_feature

        return normalized_data
        
    def get_data_arr(self, file: str) -> Optional[np.ndarray]:
        file = self._get_path(file)
        df = pd.read_csv(file)
        data = df.iloc[:, 1:].replace('', np.nan).to_numpy(dtype=np.float64)

        if np.isnan(data).any() or data.shape[0] < self.sequence_length:
            return None
        
        data = data[:self.sequence_length, :]

        return data

    def _ensure_data(self):
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        if not csv_files:
            print("No data found. Downloading")
            self.downloader.download()
        else:
            print("Data found.")

    def _get_path(self, file: str) -> str:
        return os.path.join(self.data_dir, file)
    