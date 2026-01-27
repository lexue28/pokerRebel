#!/bin/bash

# Job Flags - CPU-only job with STRONGER parameters for better gameplay
#SBATCH -p mit_normal
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 10                   # CPUs per task
#SBATCH --mem=192G               # Memory (increased further for strong config)
#SBATCH --time=24:00:00         # Time limit (24 hours for stronger training)
#SBATCH --job-name=rebel_strong # Job name
#SBATCH --output=logs/logs_strong_%j.out    # Standard output
#SBATCH --error=logs/logs_strong_%j.err     # Standard error

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

# Change to project directory
cd ~/rebel

# Create logs directory if it doesn't exist
mkdir -p logs

# STRONGER config for better gameplay:
# - num_iters=64: More CFR iterations per subgame (was 16)
# - max_depth=3: Deeper subgame trees (was 2)
# - cpu_gen_threads=2: Reduced from 4 to lower peak memory (4 subgames in parallel = 4x memory)
# - replay.capacity=40000: Slightly reduced to save memory (was 50000)
# - data.train_epoch_size=3200: More training per epoch (was 200)
# - data.train_batch_size=128: Larger batches (was 16)
# This will produce a MUCH stronger model for gameplay
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml device=cpu exploit=false selfplay.cpu_gen_threads=2 replay.capacity=40000 data.train_epoch_size=3200 data.train_batch_size=128 env.subgame_params.num_iters=64 env.subgame_params.max_depth=3
