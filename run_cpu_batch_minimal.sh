#!/bin/bash

# Job Flags - CPU-only job with MINIMAL parameters (fastest, lowest memory)
# Use this to avoid partition time limits - finishes faster
#SBATCH -p mit_normal
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=200G               # High memory (nodes have 385G)
#SBATCH --time=8:00:00          # Shorter time to avoid partition limit
#SBATCH --job-name=rebel_minimal
#SBATCH --output=logs/logs_minimal_%j.out
#SBATCH --error=logs/logs_minimal_%j.err

module load miniforge
ENV_PATH="/home/lexue28/miniconda3/envs/rebel_cpu"
export PATH="${ENV_PATH}/bin:$PATH"
export CONDA_DEFAULT_ENV=rebel_cpu
export LD_LIBRARY_PATH="${ENV_PATH}/lib:$LD_LIBRARY_PATH"
module load gcc 2>/dev/null || echo "GCC module not available"

cd ~/rebel
mkdir -p logs

# MINIMAL config - fastest training, lowest memory:
# - num_iters=24: Lower but still decent quality
# - max_depth=3: Keep at 3 (critical for quality)
# - cpu_gen_threads=1: Single thread
# - replay.capacity=20000: Minimal buffer
# - max_epochs=500: Fewer epochs to finish faster
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    device=cpu \
    exploit=false \
    selfplay.cpu_gen_threads=1 \
    replay.capacity=20000 \
    data.train_epoch_size=2400 \
    data.train_batch_size=64 \
    max_epochs=500 \
    env.subgame_params.num_iters=24 \
    env.subgame_params.max_depth=3
