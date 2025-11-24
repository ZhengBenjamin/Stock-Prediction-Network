import torch
import numpy as np
import os
from typing import Optional

from models.LSTM.model import Model as LSTMModel
from models.CNN.cnn_model import CNNModel
from utils.stock_preprocessor import StockPreprocessor


class StockPredictor:
    def __init__(
        self,
        model_type: str = "lstm",
        checkpoint_path: Optional[str] = None,
        cnn_type: str = "1d",
    ):
        self.model_type = model_type
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if checkpoint_path is None:
            if model_type == "cnn":
                checkpoint_path = f"checkpoints/cnn_{cnn_type}/best_model.pt"
            else:
                checkpoint_path = f"checkpoints/{model_type}/best_model.pt"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Train a model first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        input_size = checkpoint["input_size"]
        if model_type == "lstm":
            self.model = LSTMModel(input_size=input_size)
        elif model_type == "cnn":
            cnn_variant = checkpoint.get("cnn_type", cnn_type)
            if cnn_variant == "2d":
                from models.CNN.cnn_model_2d import CNN2DModel

                self.model = CNN2DModel(input_size=input_size)
            else:
                self.model = CNNModel(input_size=input_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded {model_type.upper()} model from {checkpoint_path}")
        best_loss_epoch = (
            f"Best loss: {checkpoint['loss']:.6f} "
            f"(epoch {checkpoint['epoch']})"
        )
        print(best_loss_epoch)

    def predict(
        self,
        stock_symbol: str,
        sequence_length: int = 1000,
    ) -> np.ndarray:

        preprocessor = StockPreprocessor(sequence_length=sequence_length)
        data = preprocessor.get_normalized_data()

        X = data[-1:, :, :]
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_t)
        predictions = predictions.squeeze().cpu().numpy()

        return predictions

    def predict_next_step(
        self,
        stock_symbol: str,
        sequence_length: int = 1000,
    ) -> float:
        predictions = self.predict(stock_symbol, sequence_length)
        return float(predictions[-1])


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Make predictions using trained model"
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn"],
        default="lstm",
        help="Which model to use",
    )
    parser.add_argument(
        "--stock",
        default="AAPL",
        help="Stock symbol to predict",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "next"],
        default="next",
        help="full=all predictions, next=only next step",
    )

    args = parser.parse_args()
    predictor = StockPredictor(model_type=args.model)
    if args.mode == "next":
        pred = predictor.predict_next_step(args.stock)
        print(f"\nPredicted next close price for {args.stock}: {pred:.4f}")
    else:
        preds = predictor.predict(args.stock)
        print(f"\nPredictions shape: {preds.shape}")
        print(f"First 5: {preds[:5]}")
        print(f"Last 5: {preds[-5:]}")


if __name__ == "__main__":
    main()