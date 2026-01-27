# Parameter Tuning Guide for ReBeL Poker Training

## Memory Scaling Factors

### Critical Parameters (exponential impact):
- **`max_depth`**: Memory scales ~10x per level
  - `max_depth=2`: ~2-4GB per subgame
  - `max_depth=3`: ~20-40GB per subgame
  - `max_depth=4`: ~200-400GB per subgame (too much!)

### High Impact Parameters (linear):
- **`num_iters`**: Memory scales linearly
  - `num_iters=32`: Baseline
  - `num_iters=48`: 1.5x memory
  - `num_iters=64`: 2x memory

- **`cpu_gen_threads`**: Peak memory = threads × per-thread memory
  - `cpu_gen_threads=1`: ~20-40GB peak
  - `cpu_gen_threads=2`: ~40-80GB peak
  - `cpu_gen_threads=4`: ~80-160GB peak

### Moderate Impact:
- **`replay.capacity`**: ~0.5-1GB per 10k capacity
  - `replay.capacity=25000`: ~2.5GB
  - `replay.capacity=30000`: ~3GB
  - `replay.capacity=40000`: ~4GB

- **`data.train_batch_size`**: ~100MB per 64 batch size
  - `train_batch_size=64`: ~100MB
  - `train_batch_size=128`: ~200MB

## Config Presets

### 1. Conservative (96G memory, guaranteed to work)
```bash
sbatch run_cpu_batch_conservative.sh
```
- `num_iters=32`, `max_depth=3`, `cpu_gen_threads=1`
- `replay.capacity=25000`
- **Quality**: Good (80% of strong config)
- **Speed**: Slower (single thread)

### 2. Balanced (128G memory, recommended starting point)
```bash
sbatch run_cpu_batch_balanced.sh
```
- `num_iters=48`, `max_depth=3`, `cpu_gen_threads=2`
- `replay.capacity=30000`
- **Quality**: Very good (90% of strong config)
- **Speed**: Moderate (2 threads)

### 3. Strong (192G memory, if available)
```bash
sbatch run_cpu_batch_strong.sh
```
- `num_iters=64`, `max_depth=3`, `cpu_gen_threads=2`
- `replay.capacity=40000`
- **Quality**: Excellent (100%)
- **Speed**: Fast (2 threads)

## Finding Your Edge

### Step 1: Start with Balanced
```bash
sbatch run_cpu_batch_balanced.sh
```

### Step 2: Monitor Memory Usage
```bash
# Watch job
squeue -u $USER
scontrol show job <JOBID> | grep -E "Memory|MaxRSS"

# Check logs for OOM
tail -f logs/logs_balanced_*.err | grep -i oom
```

### Step 3: If OOM, go Conservative
If balanced fails, use conservative config.

### Step 4: If Balanced Works, Try Stronger
If balanced succeeds, try increasing one parameter at a time:
- Increase `num_iters` from 48 → 56 → 64
- Increase `cpu_gen_threads` from 2 → 3 → 4
- Increase `replay.capacity` from 30000 → 35000 → 40000

## Quality vs Speed Tradeoffs

| Parameter | Quality Impact | Speed Impact | Memory Impact |
|-----------|---------------|--------------|---------------|
| `num_iters` | High (more accurate) | Low (slightly slower) | Medium |
| `max_depth` | Very High (much better) | Medium (slower) | Very High |
| `cpu_gen_threads` | None | High (faster) | High |
| `replay.capacity` | Low (more diverse data) | None | Medium |
| `train_batch_size` | Low (more stable) | Low (faster training) | Low |

## Recommended Strategy

1. **Start**: Balanced config (128G, num_iters=48, max_depth=3, threads=2)
2. **If OOM**: Conservative config (96G, num_iters=32, threads=1)
3. **If Success**: Gradually increase to find edge
4. **Never reduce**: `max_depth` below 3 (quality drops significantly)

## Expected Training Times

With `max_depth=3`:
- Conservative: ~2-3 hours per epoch
- Balanced: ~1.5-2 hours per epoch  
- Strong: ~1-1.5 hours per epoch

Warmup (256 games):
- Conservative: ~2-3 hours
- Balanced: ~1.5-2 hours
- Strong: ~1-1.5 hours
