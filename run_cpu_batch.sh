#!/bin/bash

# Job Flags - CPU-only job on Engaging portal
#SBATCH -p mit_normal
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 10                   # CPUs per task (8 for generation + 2 for training overhead)
#SBATCH --mem=16G               # Memory (16GB for neural network + replay buffer)
#SBATCH --time=48:00:00         # Time limit (48 hours for 3000 epochs)
# Note: For liars_dice with cpu_gen_threads=60, increase -c to 64 and --mem to 32G
#SBATCH --job-name=rebel_cpu    # Job name
#SBATCH --output=logs/logs_%j.out    # Standard output
#SBATCH --error=logs/logs_%j.err     # Standard error

# Set up environment
module load miniforge

# Activate conda environment (adjust environment name if different)
source activate rebel
# Or if using conda activate:
# conda activate rebel

# Change to project directory (adjust path if needed)
cd ~/rebel

# Create logs directory if it doesn't exist
mkdir -p logs

# Run your application with CPU-only configuration
# Using --adhoc flag and setting device=cpu, launcher=local to avoid submitit
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    device=cpu \
    launcher=local \
    selfplay.cpu_gen_threads=8
