import os
import pandas as pd
import numpy as np
import copy

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from .download_data import DownloadData

from typing import List, Optional, LiteralString

class StockPreprocessor:
    def __init__(self, data_dir: Optional[str] = "data", test_size: Optional[float] = 0.2, sequence_length: Optional[int] = 10):
        self.downloader = DownloadData()
        self.data_dir = data_dir
        self.test_size = test_size
        self.sequence_length = sequence_length
        self._ensure_data()

    def parse_data(self) -> np.array:
        """Parses each CSV file into a list of np.array
        Excludes attr labels + timestamps
        Limits to 1000 data points per stock"""

        stock_data = []

        for file in os.listdir(self.data_dir):
            data = self.get_data_arr(file)

            if data is not None:
                stock_data.append(data)
        
        return np.array(stock_data)
    
    def normalize(self, stock_data: np.array) -> np.array:
        """Normalizes data matrix """

        attr_dict = defaultdict(list) # attr_index : List[attr_vals]
        normalized_attr_dict = dict()  # attr_index : List[norm_attr_vals]

        # Convert data into attr_dict
        for stock in stock_data:
            for data_pt in stock:
                for attr_index, attr_val in enumerate(data_pt):
                    attr_dict[attr_index].append(attr_val)

        # Normalize
        for attr_index, attr_val_list in attr_dict.items():
            attr_val_list = np.array(attr_val_list)
            scaler = MinMaxScaler()
            normalized_attr_dict[attr_index] = scaler.fit_transform(attr_val_list.reshape(-1, 1))

        
        
            
        
        
    def get_data_arr(self, file: str) -> np.array:
        """Takes file and converts to numpy array without timestamps, 
        returns None if there is missing data"""

        file = self._get_path(file)
        df = pd.read_csv(file)
        data = df.iloc[:, 1:].replace('', np.nan).to_numpy(dtype=np.float64)

        if np.isnan(data).any() or data.shape[0] < 1000:
            return None
        
        data = data[:1000, :]

        return data

    def _ensure_data(self):
        """Check if data is there, otherwise download it"""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        if not csv_files:
            print("No data found. Downloading")
            self.downloader.download()
        else:
            print("Data found.")

    def _get_path(self, file: str) -> LiteralString:
        """Returns path of file"""
        return os.path.join(self.data_dir, file)
    