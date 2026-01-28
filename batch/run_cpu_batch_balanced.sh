#!/bin/bash

# Job Flags - CPU-only job with balanced parameters
# This config balances model quality with resource usage
#SBATCH -p mit_normal
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 10                   # CPUs per task
#SBATCH --mem=300G              # High memory (nodes have 385G)
#SBATCH --time=12:00:00         # Time limit (reduced to avoid partition limit)
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

# BALANCED config - moderate speedup to reach epoch 10:
# - num_iters=40: Reduced from 48 (~17% faster, minimal quality loss)
# - max_depth=3: Keep at 3 (critical for quality)
# - cpu_gen_threads=2: Keep same (balance between speed and memory)
# - replay.capacity=25000: Reduced from 30000 (~17% faster buffer ops)
# - data.train_epoch_size=1000: Reduced from 1200 (~17% faster epochs)
# - data.train_batch_size=96: Reduced from 128 (~25% faster training)
# Target: ~1 hour/epoch to reach epoch 10 in 12 hours
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    device=cpu \
    exploit=false \
    selfplay.cpu_gen_threads=2 \
    replay.capacity=25000 \
    data.train_epoch_size=1000 \
    data.train_batch_size=96 \
    env.subgame_params.num_iters=40 \
    env.subgame_params.max_depth=3
