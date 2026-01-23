#!/bin/bash

# Job Flags - CPU-only job on Engaging portal
#SBATCH -p mit_normal
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 5                    # CPUs per task (reduced for testing)
#SBATCH --mem=16G               # Memory (16GB for neural network + replay buffer)
#SBATCH --time=2:00:00          # Time limit (2 hours for testing)
# Note: For liars_dice with cpu_gen_threads=60, increase -c to 64 and --mem to 32G
#SBATCH --job-name=rebel        # Job name
#SBATCH --output=logs/logs_%j.out    # Standard output
#SBATCH --error=logs/logs_%j.err     # Standard error

# Set up environment
module load miniforge

# Initialize conda (needed for conda activate to work in batch scripts)
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate rebel_cpu

# Verify we're in the right environment and Python path
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Verify pytorch_lightning is available
python -c "import pytorch_lightning; print('pytorch_lightning import successful')" || echo "ERROR: pytorch_lightning not found!"

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
