# Slurm Cluster Setup Guide

## Overview
ReBeL uses `submitit` to automatically handle Slurm job submission. **You don't need to write manual `sbatch` scripts!** Just run the Python command and it will submit the job for you.

## Step 1: Transfer Code to Cluster

From your local machine (Windows), transfer the code to the cluster:

```bash
# On your local machine (PowerShell or Git Bash)
scp -r C:\Users\lexue\OneDrive\ComputerProjects\MIT\rebel lexue28@login007.mit.edu:~/rebel
```

Or use `rsync` for better efficiency:
```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  C:\Users\lexue\OneDrive\ComputerProjects\MIT\rebel/ \
  lexue28@login007.mit.edu:~/rebel/
```

## Step 2: SSH into Cluster and Set Up Environment

```bash
ssh lexue28@login007.mit.edu
cd ~/rebel
```

## Step 3: Set Up Conda Environment

The cluster might have conda in a different location. Try these:

```bash
# Option 1: Check if conda is in your PATH
which conda

# Option 2: Try common locations
source ~/.bashrc
# or
source /etc/profile.d/conda.sh
# or
export PATH="$HOME/miniconda3/bin:$PATH"
# or
export PATH="/opt/conda/bin:$PATH"

# Then create environment
conda create --yes -n rebel python=3.7
conda activate rebel
```

If conda isn't available, ask your cluster admin or check:
```bash
module avail  # See what modules are available
module load anaconda  # or similar
```

## Step 4: Install Dependencies

```bash
# Make sure you're in the rebel conda environment
conda activate rebel

# Install Python dependencies
pip install -r requirements.txt

# Install cmake (needed for C++ compilation)
conda install cmake -y
# OR if conda doesn't have it:
# module load cmake  # Check with module avail first
```

## Step 5: Build C++ Code

```bash
# Make sure you're in the project directory
cd ~/rebel

# Clean and build
make clean
make

# If make fails, you might need to load compiler modules:
# module load gcc
# module load cmake
```

## Step 6: Configure Slurm Partition

The default `slurm.yaml` doesn't specify a partition. You need to add your partition name.

**Option A: Override in command (recommended)**
```bash
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    launcher=slurm \
    launcher.partition=sched_mit_kburdge_r8 \
    launcher.num_gpus=2 \
    launcher.hours=72 \
    launcher.mem_per_gpu=62 \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=0 \
    selfplay.threads_per_gpu=8
```

**Option B: Create custom launcher config**
Create `conf/common/launcher/slurm_mit.yaml`:
```yaml
launcher:
  driver: slurm
  partition: sched_mit_kburdge_r8
  num_gpus: 2
  cpus_per_gpu: 80
  hours: 72
  volta32: false
  mem_per_gpu: 62
  single_task_per_node: false
```

Then use: `launcher=slurm_mit`

## Step 7: Run Training Job

```bash
# Make sure conda environment is activated
conda activate rebel

# Run the command (it will automatically submit to Slurm)
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    launcher=slurm \
    launcher.partition=sched_mit_kburdge_r8 \
    launcher.num_gpus=2 \
    launcher.hours=72 \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=0 \
    selfplay.threads_per_gpu=8
```

**What happens:**
- The code detects you're NOT already on a Slurm node (you're on login node)
- It uses `submitit` to create a Slurm job
- It submits the job and prints the job ID
- You'll see output like:
  ```
  Submitted job 12345
  stdout: tail -F exp_outputs/slurm/12345_0_log.out
  ```

## Step 8: Monitor Job

```bash
# Check job status
squeue -u lexue28

# View logs (replace JOB_ID with actual job ID from step 7)
tail -F exp_outputs/slurm/JOB_ID_0_log.out

# Or check stderr
tail -F exp_outputs/slurm/JOB_ID_0_log.err
```

## Troubleshooting

### "Invalid account or account/partition combination"
- You might need to specify an account. Check with:
  ```bash
  sacctmgr show user lexue28
  ```
- Add account to command: `launcher.account=YOUR_ACCOUNT`

### "conda: command not found"
- Try loading conda module: `module load anaconda` or `module load conda`
- Or add conda to PATH manually (see Step 3)

### "cuda/12.0: unknown module"
- ReBeL handles CUDA automatically via PyTorch
- You don't need to manually load CUDA modules
- Just make sure PyTorch is installed (via `requirements.txt`)

### Build errors
- Make sure you have compiler modules loaded:
  ```bash
  module load gcc
  module load cmake
  ```

### "No GPUs available"
- Check GPU availability: `sinfo -p sched_mit_kburdge_r8 -o "%N %G"`
- Make sure you're requesting GPUs: `launcher.num_gpus=2`
- Check if you need `--gres=gpu:2` format (ReBeL handles this automatically)

## Quick Reference

**Key files:**
- `conf/c02_selfplay/poker.yaml` - Training config
- `conf/common/launcher/slurm.yaml` - Default Slurm settings
- `exp_outputs/` - Where logs and checkpoints go

**Important:**
- Run from login node (not compute node)
- ReBeL automatically submits to Slurm
- No manual `sbatch` script needed!
- Check logs in `exp_outputs/slurm/` directory
