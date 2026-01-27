# Fixing PartitionTimeLimit Error

## What PartitionTimeLimit Means

"PartitionTimeLimit" means you've exceeded the **total time** you can have jobs running/queued in the `mit_normal` partition. This is a **partition policy**, not a memory issue.

## Solutions (in order of preference):

### 1. Cancel Old Pending Jobs (FASTEST FIX)

```bash
# Cancel all pending jobs
squeue -u $USER | grep PD | awk '{print $1}' | xargs scancel

# Or cancel specific jobs
scancel 8454249 8454222 8454311
```

This frees up your "time budget" immediately.

### 2. Use Interactive Session (salloc) Instead

Interactive sessions often have different limits:

```bash
# Request interactive node
salloc -p mit_normal -n 1 -c 10 --mem=96G --time=12:00:00

# Once you get the node, run directly:
cd ~/rebel
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    device=cpu exploit=false \
    selfplay.cpu_gen_threads=1 \
    replay.capacity=25000 \
    data.train_epoch_size=3200 \
    data.train_batch_size=128 \
    env.subgame_params.num_iters=32 \
    env.subgame_params.max_depth=3
```

**Advantages:**
- Different time limits (often more lenient)
- Can monitor in real-time
- Can stop early if needed

### 3. Use Shorter Time Limits

Reduce `--time` in batch scripts:
- `--time=24:00:00` → `--time=12:00:00` (already done in conservative)
- `--time=12:00:00` → `--time=8:00:00` (minimal config)

### 4. Use Minimal Config (Fastest Training)

```bash
sbatch run_cpu_batch_minimal.sh
```

This uses:
- Lower `num_iters=24` (faster)
- Lower `replay.capacity=20000` (less memory)
- Fewer `max_epochs=500` (finishes faster)
- Shorter `--time=8:00:00`

### 5. Check Partition Limits

```bash
# See partition limits
sinfo -p mit_normal -o "%P %l %L %D %T"

# Check your current usage
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

### 6. Wait for Jobs to Complete

If you have running jobs, wait for them to finish. The time limit resets after jobs complete.

## Recommended Action Plan

**Step 1: Cancel pending jobs**
```bash
squeue -u $USER | grep PD | awk '{print $1}' | xargs scancel
```

**Step 2: Use minimal config for quick test**
```bash
sbatch run_cpu_batch_minimal.sh
```

**Step 3: If minimal works, scale up gradually**
- Start with minimal
- If it succeeds, try conservative
- Then balanced, then strong

## About GPU Training

**Yes, GPU training works!** But you need:
1. CUDA-enabled PyTorch (you have CPU-only)
2. GPU partition access
3. Different config

To use GPU:
```bash
# Change device to cuda in config
device=cuda

# Use GPU for data generation
selfplay.cpu_gen_threads=0
selfplay.threads_per_gpu=8
```

But you'd need to rebuild PyTorch with CUDA support first.

## Quick Fix Right Now

```bash
# 1. Cancel all pending jobs
squeue -u $USER | grep PD | awk '{print $1}' | xargs scancel

# 2. Use minimal config (fastest, avoids partition limit)
sbatch run_cpu_batch_minimal.sh

# 3. Monitor
watch -n 5 "squeue -u $USER"
```
