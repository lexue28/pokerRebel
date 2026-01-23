#!/bin/bash

# Job Flags - CPU-only job on Engaging portal
#SBATCH -p mit_normal
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 10                   # CPUs per task (8 for generation + 2 for training overhead)
#SBATCH --mem=128G              # Memory (128GB - increased due to OOM with 64GB)
#SBATCH --time=10:00:00         # Time limit (10 hours for CPU training)
# Note: For liars_dice with cpu_gen_threads=60, increase -c to 64 and --mem to 64G
#SBATCH --job-name=rebel        # Job name
#SBATCH --output=logs/logs_%j.out    # Standard output
#SBATCH --error=logs/logs_%j.err     # Standard error

# Set up environment
module load miniforge

# Use direct path to conda environment (more reliable in batch scripts)
ENV_PATH="/home/lexue28/miniconda3/envs/rebel_cpu"
export PATH="${ENV_PATH}/bin:$PATH"
export CONDA_DEFAULT_ENV=rebel_cpu

# Fix GLIBCXX issue: Use conda's libstdc++ instead of system one
export LD_LIBRARY_PATH="${ENV_PATH}/lib:$LD_LIBRARY_PATH"

# Try to load GCC module if available (for newer libstdc++)
module load gcc 2>/dev/null || echo "GCC module not available, using conda's libstdc++"

echo "=== Environment Setup ==="
echo "Using Python from: ${ENV_PATH}/bin/python"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Verify pytorch_lightning is available
echo "=== Testing pytorch_lightning import ==="
python -c "import pytorch_lightning; print('SUCCESS: pytorch_lightning import successful')" || echo "ERROR: pytorch_lightning not found!"
echo ""

# Change to project directory (adjust path if needed)
cd ~/rebel

# Create logs directory if it doesn't exist
mkdir -p logs

# Run your application with CPU-only configuration
# Using --adhoc flag and setting device=cpu (no launcher = runs locally)
# Aggressively reduced memory usage for poker (22,100 hands = large samples)
# If still OOM, reduce replay.capacity to 50000 or 25000
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml device=cpu selfplay.cpu_gen_threads=8 replay.capacity=50000 data.train_epoch_size=3200 data.train_batch_size=128
