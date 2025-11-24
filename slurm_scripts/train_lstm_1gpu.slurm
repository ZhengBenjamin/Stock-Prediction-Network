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

echo "LSTM Single GPU Training"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source /myenv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python main.py \
    --mode train \
    --model lstm \
    --epochs 1000 \
    --batch_size 64 \
    --device cuda

echo "Training completed at: $(date)"
