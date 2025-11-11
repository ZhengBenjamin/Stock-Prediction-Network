import argparse

# Import trainers explicitly
from models.LSTM.trainer import Trainer as LSTMTrainer
from models.CNN.trainer import CNNTrainer


def main():

    parser = argparse.ArgumentParser(
        description="Train a stock prediction model"
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn"],
        default="lstm",
        help="Which model to train (default: lstm)",
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

    args = parser.parse_args()

    if args.model == "lstm":
        trainer = LSTMTrainer(batch_size=args.batch_size)
    else:
        trainer = CNNTrainer(batch_size=args.batch_size)

    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
