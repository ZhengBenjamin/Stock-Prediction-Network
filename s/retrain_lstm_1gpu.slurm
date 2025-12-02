#!/bin/bash
#SBATCH --job-name=lstm_1gpu_quick
#SBATCH --account=csds438
#SBATCH --output=logs/lstm_1gpu_retrain_%j.out
#SBATCH --error=logs/lstm_1gpu_retrain_%j.err
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00

echo "Retraining LSTM 1 GPU"
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /mnt/vstor/courses/csds438/pxg388/Stock_Prediction_Project/myenv/bin/activate

python main.py --model lstm --epochs 1000 --batch_size 64

echo "Done: $(date)"
