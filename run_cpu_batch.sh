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

# Debug: List available environments
echo "=== Available conda environments ==="
conda info --envs
echo ""

# Activate conda environment (rebel_cpu)
echo "Activating rebel_cpu environment..."
conda activate rebel_cpu || {
    echo "conda activate failed, trying source activate..."
    source activate rebel_cpu || {
        echo "Both activation methods failed, using direct path..."
        export PATH="/home/lexue28/miniconda3/envs/rebel_cpu/bin:$PATH"
        export CONDA_DEFAULT_ENV=rebel_cpu
    }
}

# Verify activation
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "rebel_cpu" ]; then
    echo "WARNING: Environment may not be activated correctly"
    echo "Using Python from: /home/lexue28/miniconda3/envs/rebel_cpu/bin/python"
    export PATH="/home/lexue28/miniconda3/envs/rebel_cpu/bin:$PATH"
fi
echo "Active environment: ${CONDA_DEFAULT_ENV:-rebel_cpu (via PATH)}"

# Verify we're in the right environment and Python path
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
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
# Using --adhoc flag and setting device=cpu, launcher=local to avoid submitit
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    device=cpu \
    launcher=local \
    selfplay.cpu_gen_threads=8
