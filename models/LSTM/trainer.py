import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import Model
from .stock_dataloader import StockDataloader
from .benchmark import ModelBenchmark
from utils.stock_preprocessor import StockPreprocessor
from typing import Optional

class Trainer:
    def __init__(self, 
                 hidden_size: Optional[int] = 64, 
                 num_layers: Optional[int] = 2, 
                 dropout: Optional[float] = 0.2,
                 sequence_length: Optional[int] = 1000,
                 batch_size: Optional[int] = 32,
                 lr: Optional[float] = 0.001,
                 device: Optional[str] = "cuda"):
        
        self.preprocessor = StockPreprocessor(sequence_length=sequence_length)
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")

        self.X, self.y = self._prepare_data()
        print(f"Data shape - X: {self.X.shape}, y: {self.y.shape}")
        self.dataset = StockDataloader(self.X, self.y)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.model = Model(
            input_size=self.X.shape[2], 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout
        ).to(self.device)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        self.benchmark = ModelBenchmark(
            model=self.model,
            preprocessor=self.preprocessor,
            device=str(self.device)
        )

    def train(self, epochs: Optional[int] = 10):
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (X_batch, y_batch) in enumerate(self.loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.loader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Avg Loss: {avg_loss:.6f}")
        
        print(f"Training completed! Final loss: {avg_loss:.6f}")

    def run_benchmark(self, plot: bool = True):
        print("Running Benchmark Evaluations")
        
        results = self.benchmark.evaluate(
            X=self.X,
            y=self.y,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        self.benchmark.print_metrics()
        if plot:
            self.benchmark.plot_predictions(max_points=500)
            self.benchmark.plot_residuals()
        
        return results
    
    def _prepare_data(self):
        data = self.preprocessor.get_normalized_data()
        num_stocks, seq_len, num_features = data.shape
        X = data[:, :-1, :]
        y = data[:, -1, 4]
        print(f"Prepared data - X: {X.shape}, y: {y.shape}")
        return X, y
