# Speed vs Quality Trade-offs

## Current Situation
- **Current speed**: ~3 hours per epoch (at epoch 4 after 12 hours)
- **Goal**: Reach epoch 10 within 12 hours
- **Problem**: QOSMaxMemoryPerUser when running 2 jobs

## Recommendation: **Cancel and Restart with Fast Config**

### Why Cancel?
1. At current speed (3 hrs/epoch), you'll only reach epoch 5-6 in 12 hours
2. You need epoch 10, which would take ~30 hours at current speed
3. Fast config can get you to epoch 10 in ~8-10 hours

### Fast Config Benefits
- **num_iters=32** (vs 48): 33% faster subgame solving, minimal quality loss
- **train_epoch_size=800** (vs 1200): 33% faster epochs
- **train_batch_size=64** (vs 128): Faster training steps
- **replay.capacity=20000** (vs 30000): Faster buffer operations
- **mem=250G** (vs 300G): Lower memory to avoid QOS limits

### Expected Speed
- **Before**: ~3 hours/epoch → 4 epochs in 12 hours
- **After**: ~1-1.5 hours/epoch → 8-10 epochs in 12 hours ✅

## Action Plan

1. **Cancel current job**:
   ```bash
   squeue -u $USER  # Find job ID
   scancel <job_id>
   ```

2. **Use fast config**:
   ```bash
   sbatch run_cpu_batch_fast.sh
   ```

3. **Monitor progress**:
   ```bash
   tail -f logs/logs_fast_<job_id>.out
   ```

4. **Extract epoch 10 when done**:
   ```bash
   python extract_epoch.py 10
   ```

## Quality Impact

The fast config will have:
- **Slightly lower quality** than balanced config
- But **still usable** for testing your bot
- You can always retrain with better params later

**Bottom line**: Get to epoch 10 now with fast config, then decide if you need better quality later.
