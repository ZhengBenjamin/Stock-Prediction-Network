import numpy as np

from sklearn.model_selection import train_test_split
from model import *
from dataloader import *
from utils.stock_preprocessor import StockPreprocessor

from typing import Optional, List

class Trainer:
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: Optional[int] = 64, 
                 num_layers: Optional[int] = 2, 
                 dropout: Optional[int] = 0.2):
        
        self.model = Model(input_size, hidden_size, num_layers, dropout)
        self.preprocessor = StockPreprocessor()

    def train(self, epochs: Optional[int] = 100000):
        pass 