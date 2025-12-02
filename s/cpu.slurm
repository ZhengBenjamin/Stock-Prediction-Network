#!/bin/bash
#SBATCH --job-name=train_all_cpu
#SBATCH --account=csds438
#SBATCH --output=logs/train_all_cpu_%j.out
#SBATCH --error=logs/train_all_cpu_%j.err
#SBATCH --partition=markov_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "=========================================="
echo "CPU Training - All Models"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs Allocated: $SLURM_CPUS_PER_TASK"
echo "=========================================="

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /mnt/vstor/courses/csds438/pxg388/Stock_Prediction_Project/myenv/bin/activate

# Display CPU information
echo "CPU Information:"
lscpu | grep "Model name"
echo ""

# Train LSTM model on CPU
echo "=========================================="
echo "Training LSTM Model on CPU"
echo "Start: $(date)"
echo "=========================================="

python main.py \
    --mode train \
    --model lstm \
    --epochs 1000 \
    --batch_size 64 \
    --device cpu

echo "LSTM training completed at: $(date)"
echo ""

# Train CNN 1D model on CPU
echo "=========================================="
echo "Training CNN 1D Model on CPU"
echo "Start: $(date)"
echo "=========================================="

python main.py \
    --mode train \
    --model cnn \
    --cnn_type 1d \
    --epochs 1000 \
    --batch_size 64 \
    --device cpu

echo "CNN 1D training completed at: $(date)"
echo ""

# Train CNN 2D model on CPU
echo "=========================================="
echo "Training CNN 2D Model on CPU"
echo "Start: $(date)"
echo "=========================================="

python main.py \
    --mode train \
    --model cnn \
    --cnn_type 2d \
    --epochs 1000 \
    --batch_size 64 \
    --device cpu

echo "CNN 2D training completed at: $(date)"
echo ""

echo "=========================================="
echo "All CPU training completed at: $(date)"
echo "=========================================="
