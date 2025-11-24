import torch
import os
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from models.CNN.cnn_model import CNNModel
from models.LSTM.stock_dataloader import StockDataloader
from utils.stock_preprocessor import StockPreprocessor

class CNNTrainer:
    # Trainer function for using the CNNModel with sequence-to-sequence handling.
    def __init__(
        self,
        hidden_channels: Optional[int] = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        sequence_length: Optional[int] = 1000,
        batch_size: Optional[int] = 32,
        lr: Optional[float] = 1e-3,
        cnn_type: str = "1d",
    ):
        self.preprocessor = StockPreprocessor(sequence_length=sequence_length)
        self.lr = lr
        self.batch_size = batch_size
        self.cnn_type = cnn_type

        self.X, self.y = self._prepare_data()
        
        self.data = CNNDataset(self.X, self.y)
        self.loader = DataLoader(self.data, batch_size, shuffle=True)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        input_size = self.X.shape[2]

        if self.cnn_type == "2d":
            from models.CNN.cnn_model_2d import CNN2DModel
            self.model = CNN2DModel(
                input_size=input_size,
                hidden_channels=hidden_channels,
                kernel_time=kernel_size,
                kernel_feat=kernel_size,
                dropout=dropout,
            ).to(self.device)
        else:
            self.model = CNNModel(
                input_size=input_size,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            ).to(self.device)

        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

        self.checkpoint_dir = f"checkpoints/cnn_{self.cnn_type}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_loss = float('inf')

    def train(self, epochs: Optional[int] = 1000):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"\nStarting training for {epochs} epochs...\n")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for X_batch, y_batch in self.loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.loader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Avg Loss: {avg_loss:.6f}")

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, "best_model.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "loss": avg_loss,
                        "input_size": self.X.shape[2],
                        "cnn_type": self.cnn_type,
                    },
                    checkpoint_path,
                )
                if epoch % 10 == 0 or (self.best_loss - avg_loss) > 0.00001:
                    print(f"Checkpoint saved to {checkpoint_path}")

        print(f"Training completed! Final loss: {avg_loss:.6f}")

    def _prepare_data(self) -> Tuple:
        data = self.preprocessor.get_normalized_data()
        X = data[:, :-1, :]
        y = data[:, 1:, 4:5]
        print(f"Prepared data - X: {X.shape}, y: {y.shape}")
        return X, y


class CNNDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        print(f"Dataset created: X shape = {self.X.shape}, y shape = {self.y.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
