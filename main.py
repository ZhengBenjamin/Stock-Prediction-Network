import argparse
import torch
from models.LSTM.trainer import Trainer as LSTMTrainer
from models.CNN.trainer import CNNTrainer
from inference import StockPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Train or run inference on stock prediction model"
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "inference"],
        default="train",
        help="train: train a model, predict/inference: use trained model",
    )
    
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn"],
        default="lstm",
        help="Which model to use (lstm or cnn)",
    )
    
    parser.add_argument(
        "--cnn_type",
        choices=["1d", "2d"],
        default="1d",
        help="CNN variant to use (only for --model cnn)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for computation",
    )
    parser.add_argument(
        "--stock",
        default="AAPL",
        help="Stock symbol for prediction (inference mode only)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (for inference)",
    )

    args = parser.parse_args()

    model_desc = f"{args.model.upper()}"
    if args.model == "cnn":
        model_desc += f" ({args.cnn_type.upper()})"
    print(f"MODE: {args.mode.upper()} | MODEL: {model_desc}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Device: {args.device}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")

    if args.mode == "train":
        print(f"Training {model_desc} model...")
        print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}\n")
        
        if args.model == "lstm":
            trainer = LSTMTrainer(
                batch_size=args.batch_size,
                lr=args.lr
            )
        else:
            trainer = CNNTrainer(
                batch_size=args.batch_size,
                lr=args.lr,
                cnn_type=args.cnn_type
            )
        trainer.train(args.epochs)
        
    else:
        print(f"Running inference with {model_desc} model...")
        predictor = StockPredictor(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            cnn_type=args.cnn_type
        )
        pred = predictor.predict_next_step(args.stock)
        print(f"Predicted next close price for {args.stock}: {pred:.6f}")

if __name__ == "__main__":
    main()
