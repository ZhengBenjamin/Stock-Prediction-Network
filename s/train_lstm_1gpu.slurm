#!/bin/bash
#SBATCH --job-name=lstm_1gpu
#SBATCH --account=csds438
#SBATCH --output=logs/lstm_1gpu_%j.out
#SBATCH --error=logs/lstm_1gpu_%j.err
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "=========================================="
echo "LSTM Single GPU Training"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate virtual environment
source /mnt/vstor/courses/csds438/pxg388/Stock_Prediction_Project/myenv/bin/activate

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run training
python main.py \
    --mode train \
    --model lstm \
    --epochs 1000 \
    --batch_size 64 \
    --device cuda

echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
