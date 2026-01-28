#!/bin/bash

# Job Flags - CPU-only job with FAST parameters (prioritize speed to reach epoch 10)
# This config trades some quality for speed to ensure we reach epoch 10 in 12 hours
#SBATCH -p mit_normal
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=250G              # Reduced from 300G to avoid QOS limits
#SBATCH --time=12:00:00
#SBATCH --job-name=rebel_fast
#SBATCH --output=logs/logs_fast_%j.out
#SBATCH --error=logs/logs_fast_%j.err

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

# FAST config - prioritize speed to reach epoch 10:
# - num_iters=32: Reduced from 48 (faster subgame solving)
# - max_depth=3: Keep at 3 (critical for quality, but 3 is minimum)
# - cpu_gen_threads=2: Keep same (balance between speed and memory)
# - replay.capacity=20000: Reduced from 30000 (faster buffer operations)
# - data.train_epoch_size=800: Reduced from 1200 (faster epochs)
# - data.train_batch_size=64: Reduced from 128 (faster training steps)
# - max_epochs=15: Just enough to get to epoch 10
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    device=cpu \
    exploit=false \
    selfplay.cpu_gen_threads=2 \
    replay.capacity=20000 \
    data.train_epoch_size=800 \
    data.train_batch_size=64 \
    max_epochs=15 \
    env.subgame_params.num_iters=32 \
    env.subgame_params.max_depth=3
