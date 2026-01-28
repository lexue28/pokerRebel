#!/bin/bash

# Job Flags - CPU-only job with CONSERVATIVE parameters (safest)
# Use this if balanced config still OOMs
#SBATCH -p mit_normal
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=250G               # High memory (nodes have 385G)
#SBATCH --time=12:00:00  # Reduced time limit to avoid partition limit
#SBATCH --job-name=rebel_conservative
#SBATCH --output=logs/logs_conservative_%j.out
#SBATCH --error=logs/logs_conservative_%j.err

module load miniforge
ENV_PATH="/home/lexue28/miniconda3/envs/rebel_cpu"
export PATH="${ENV_PATH}/bin:$PATH"
export CONDA_DEFAULT_ENV=rebel_cpu
export LD_LIBRARY_PATH="${ENV_PATH}/lib:$LD_LIBRARY_PATH"
module load gcc 2>/dev/null || echo "GCC module not available"

cd ~/rebel
mkdir -p logs

# CONSERVATIVE config - guaranteed to work:
# - num_iters=32: Moderate quality
# - max_depth=3: Keep at 3 (important)
# - cpu_gen_threads=1: Single thread (lowest memory)
# - replay.capacity=25000: Further reduced
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    device=cpu \
    exploit=false \
    selfplay.cpu_gen_threads=1 \
    replay.capacity=25000 \
    data.train_epoch_size=1200 \
    data.train_batch_size=128 \
    env.subgame_params.num_iters=32 \
    env.subgame_params.max_depth=3
