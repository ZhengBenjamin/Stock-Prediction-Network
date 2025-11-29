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
        # Store per-feature scalers for denormalization
        self.feature_scalers: List[MinMaxScaler] = []
        self._ensure_data()

    def get_normalized_data(self) -> np.ndarray:
        """Gets normalized data directly"""

        if self.normalized_data is not None:
            return self.normalized_data
        
        data = self.parse_data()
        normalized_data = self.normalize(data)
        self.normalized_data = normalized_data
        return normalized_data

    def parse_data(self) -> np.ndarray:
        """Parses each CSV file into a list of np.array
        Excludes attr labels + timestamps
        Limits to sequence_length data points per stock
        Returns np.array(stock, stock_examples, attributes)"""

        stock_data = []

        for file in os.listdir(self.data_dir):
            data = self.get_data_arr(file)

            if data is not None:
                stock_data.append(data)
        
        return np.array(stock_data)
    
    def get_close_val(self, stock_idx: int, time_idx: int) -> float:
        """Gets the close value of a particular stock at a particular time index"""
        
        if self.normalized_data is None:
            self.get_normalized_data()
        
        return self.normalized_data[stock_idx, time_idx, 4]
    
    def normalize(self, stock_data: np.ndarray) -> np.ndarray:
        """Normalizes data matrix"""

        normalized_data = np.zeros_like(stock_data)
        num_features = stock_data.shape[2]

        # Normalize minmax sacler
        self.feature_scalers = []
        for feature_idx in range(num_features):
            feature_vals = stock_data[:, :, feature_idx].reshape(-1, 1)
            scaler = MinMaxScaler()
            normalized_feature = scaler.fit_transform(feature_vals).reshape(
                stock_data.shape[0], stock_data.shape[1]
            )
            normalized_data[:, :, feature_idx] = normalized_feature
            self.feature_scalers.append(scaler)

        return normalized_data

    def denormalize_close(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized close values back to original price scale.

        Accepts (dim,) or (dim, 1), returns the same shape.
        """
        if not self.feature_scalers or len(self.feature_scalers) <= 4:
            raise RuntimeError("Scalers are not fitted yet. Call get_normalized_data() first.")

        original_shape = values.shape
        vals = values.reshape(-1, 1)
        inv = self.feature_scalers[4].inverse_transform(vals)
        return inv.reshape(original_shape)
        
    def get_data_arr(self, file: str) -> Optional[np.ndarray]:
        """Takes file and converts to numpy array without timestamps, 
        returns None if there is missing data"""

        file = self._get_path(file)
        df = pd.read_csv(file)
        data = df.iloc[:, 1:].replace('', np.nan).to_numpy(dtype=np.float64)

        if np.isnan(data).any() or data.shape[0] < self.sequence_length:
            return None
        
        data = data[:self.sequence_length, :]

        return data

    def _ensure_data(self):
        """Check if data is there, otherwise download it"""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        if not csv_files:
            print("No data found. Downloading")
            self.downloader.download()
        else:
            print("Data found.")

    def _get_path(self, file: str) -> str:
        """Returns path of file"""
        return os.path.join(self.data_dir, file)
    