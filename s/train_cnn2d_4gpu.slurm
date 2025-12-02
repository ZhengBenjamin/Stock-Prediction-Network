#!/bin/bash
#SBATCH --job-name=cnn2d_4gpu
#SBATCH --account=csds438
#SBATCH --output=logs/cnn2d_4gpu_%j.out
#SBATCH --error=logs/cnn2d_4gpu_%j.err
#SBATCH --partition=markov_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00

echo "=========================================="
echo "CNN 2D 4 GPU Training"
echo "Start: $(date)"
echo "=========================================="

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /mnt/vstor/courses/csds438/pxg388/Stock_Prediction_Project/myenv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_distributed.py \
    --model cnn \
    --cnn_type 2d \
    --epochs 1000 \
    --batch_size 64 \
    --world_size 4

echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
