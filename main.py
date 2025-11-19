import argparse

# Import trainers explicitly
from models.LSTM.trainer import Trainer as LSTMTrainer
from models.CNN.trainer import CNNTrainer
from inference import StockPredictor


def main():

    parser = argparse.ArgumentParser(
        description="Train or run inference on stock prediction model"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="train: train a model, predict: use trained model",
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn"],
        default="lstm",
        help="Which model to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--stock",
        default="AAPL",
        help="Stock symbol for prediction (predict mode only)",
    )
    parser.add_argument(
        "--cnn-type",
        choices=["1d", "2d"],
        default="1d",
        help="Which CNN variant to use when --model cnn (default: 1d)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.model == "lstm":
            trainer = LSTMTrainer(batch_size=args.batch_size)
        else:
            trainer = CNNTrainer(
                batch_size=args.batch_size, cnn_type=args.cnn_type
            )
        trainer.train(args.epochs)
    else:
        predictor = StockPredictor(
            model_type=args.model, cnn_type=args.cnn_type
        )
        pred = predictor.predict_next_step(args.stock)
        print(
            f"\nPredicted next close price for {args.stock}: "
            f"{pred:.4f}"
        )


if __name__ == "__main__":
    main()

