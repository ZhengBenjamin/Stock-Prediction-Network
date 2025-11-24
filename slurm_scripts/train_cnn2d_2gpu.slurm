#!/bin/bash
#SBATCH --job-name=cnn2d_2gpu
#SBATCH --account=csds438
#SBATCH --output=logs/cnn2d_2gpu_%j.out
#SBATCH --error=logs/cnn2d_2gpu_%j.err
#SBATCH --partition=markov_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "CNN 2D 2 GPU Training"
echo "Start: $(date)"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /myenv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_distributed.py \
    --model cnn \
    --cnn_type 2d \
    --epochs 1000 \
    --batch_size 64 \
    --world_size 2

echo "Training completed at: $(date)"
