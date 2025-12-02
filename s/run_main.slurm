#!/bin/bash
#SBATCH --job-name=main_pytorch
#SBATCH --output=main_pytorch_%j.out
#SBATCH --error=main_pytorch_%j.err
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "START PyTorch job"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# --- Load PyTorch module (includes matching Python + CUDA) ---
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Verify python
echo "Using python: $(which python)"
echo "Python version: $(python -V)"

# --- Virtual environment setup (created using module-provided python) ---
VENV_DIR=venv_pytorch

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python -m venv "$VENV_DIR" --system-site-packages
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel

    echo "Installing required Python packages..."
    pip install --no-cache-dir \
	 "typing_extensions" \
        "numpy>=1.24.0" \
        "scipy>=1.10.0" \
        "scikit-learn>=1.3.0" \
        "matplotlib>=3.7.0" \
        "seaborn>=0.12.0" \
        "pandas>=2.0.0" \
        "kagglehub>=0.3.0"
else
    echo "Using existing virtual environment $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# Optional: print CUDA visibility and torch checks
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
python - <<'PY'
import torch, sys
print("torch version:", getattr(torch, "__version__", "not-found"))
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device name:", torch.cuda.get_device_name(0))
PY

# --- Run your python program with timing like your previous script ---
MAIN_PY=main.py

echo "=============================================="
echo "Running $MAIN_PY ..."
START=$(date +%s.%N)
python "$MAIN_PY"
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)
printf "Run time: %.3f s\n" "$ELAPSED"
echo "=============================================="

# deactivate venv for cleanliness
deactivate 2>/dev/null || true

echo "End time: $(date)"
echo "END PyTorch job"

