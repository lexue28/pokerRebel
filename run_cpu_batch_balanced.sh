#!/bin/bash

# Job Flags - CPU-only job with BALANCED parameters
# This config balances model quality with resource usage
#SBATCH -p mit_normal
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 10                   # CPUs per task
#SBATCH --mem=128G              # Memory (balanced - test if this works)
#SBATCH --time=24:00:00         # Time limit
#SBATCH --job-name=rebel_balanced
#SBATCH --output=logs/logs_balanced_%j.out
#SBATCH --error=logs/logs_balanced_%j.err

# Set up environment
module load miniforge

ENV_PATH="/home/lexue28/miniconda3/envs/rebel_cpu"
export PATH="${ENV_PATH}/bin:$PATH"
export CONDA_DEFAULT_ENV=rebel_cpu
export LD_LIBRARY_PATH="${ENV_PATH}/lib:$LD_LIBRARY_PATH"
module load gcc 2>/dev/null || echo "GCC module not available"

echo "=== Environment Setup ==="
echo "Using Python from: ${ENV_PATH}/bin/python"
echo "Python version: $(python --version)"
echo ""

cd ~/rebel
mkdir -p logs

# BALANCED config - good quality without excessive memory:
# - num_iters=48: Strong but not extreme (64 was too much)
# - max_depth=3: Keep at 3 (important for quality)
# - cpu_gen_threads=2: Lower parallelism to reduce peak memory
# - replay.capacity=30000: Reduced from 40000
# - data.train_epoch_size=3200: Keep same
# - data.train_batch_size=128: Keep same
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    device=cpu \
    exploit=false \
    selfplay.cpu_gen_threads=2 \
    replay.capacity=30000 \
    data.train_epoch_size=3200 \
    data.train_batch_size=128 \
    env.subgame_params.num_iters=48 \
    env.subgame_params.max_depth=3
