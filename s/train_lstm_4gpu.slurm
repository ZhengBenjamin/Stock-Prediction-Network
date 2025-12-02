#!/bin/bash
#SBATCH --job-name=lstm_4gpu
#SBATCH --account=csds438
#SBATCH --output=logs/lstm_4gpu_%j.out
#SBATCH --error=logs/lstm_4gpu_%j.err
#SBATCH --partition=markov_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

echo "=========================================="
echo "LSTM 4 GPU Training"
echo "Start: $(date)"
echo "=========================================="

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /mnt/vstor/courses/csds438/pxg388/Stock_Prediction_Project/myenv/bin/activate

export MASTER_ADDR=localhost
export MASTER_PORT=12355

python train_distributed.py \
    --model lstm \
    --epochs 1000 \
    --batch_size 64 \
    --world_size 4

echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
