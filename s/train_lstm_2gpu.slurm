#!/bin/bash
#SBATCH --job-name=lstm_2gpu
#SBATCH --account=csds438
#SBATCH --output=logs/lstm_2gpu_%j.out
#SBATCH --error=logs/lstm_2gpu_%j.err
#SBATCH --partition=markov_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "__"
echo "LSTM 2 GPU Training"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "__"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source /mnt/vstor/courses/csds438/pxg388/Stock_Prediction_Project/myenv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "If RTX 4090 GPUs assigned, job may fail due to CUDA compatibility."
echo ""

python train_distributed.py \
    --model lstm \
    --epochs 1000 \
    --batch_size 64 \
    --world_size 2

echo "__"
echo "Training completed at: $(date)"
echo "__"
