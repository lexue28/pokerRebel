# How to Check Training Results

## Original Method (Just Check Logs)

The original ReBeL code logs everything directly. Just check the Slurm output files:

```bash
# On the cluster, check your job output
tail -f <job_id>_0_log.out    # Standard output
tail -f <job_id>_0_log.err    # Errors
```

## What the Logs Show

### During Training

You'll see logs like:

```
Setup: is_master=True n_nodes=1 rank=0 ckpt_path=.
Built a model: Net2Poker(...)
Params: [torch.float32, torch.float32, ...]
Will generate on CPU with 8 threads
Gpus for generations: ['cpu']
Threads: 8
Replay params (per buffer): {'capacity': 524288, 'seed': 10001, ...}
Train set size (forced): 12800
model size is 12345678
Creating savedir: ckpt
warming up replay buffer: 1024/1024
[Train] epoch 0 complete, avg error is 0.0234
[Train] epoch 10 complete, avg error is 0.0198
[Train] epoch 20 complete, avg error is 0.0176
Exploitability to leaf (epoch=20): 0.15
[Train] epoch 30 complete, avg error is 0.0162
...
```

### Key Log Messages

1. **Model Setup**:
   ```
   Built a model: Net2Poker(...)  # Should say Net2Poker, not Net2!
   ```

2. **Training Progress** (every epoch):
   ```
   [Train] epoch X complete, avg error is Y
   ```
   - Error should decrease over time

3. **Exploitability** (every 20 epochs if `exploit: true`):
   ```
   Exploitability to leaf (epoch=20): 0.15
   Exploitability to leaf (epoch=40): 0.12
   Exploitability to leaf (epoch=60): 0.09
   ```
   - Should decrease over time
   - Lower = better (0 = perfect)

4. **Checkpoints** (every 10 epochs):
   ```
   # Checkpoints are saved but not logged - check the ckpt/ directory
   ```

5. **Validation** (if enabled):
   ```
   [Eval] epoch 100 complete, data is valid_snapshot_0100, avg error is 0.021
   ```

6. **Data Generation**:
   ```
   time=60 items=50000 per_second=833.33
   ```
   - Shows data generation speed

7. **Final Metrics** (end of each epoch):
   ```
   Metrics: {'loss/train': 0.016, 'exploitability_last': 0.09, 'bps/gen': 0.5, ...}
   ```

## Quick Check After Job Completes

```bash
# Navigate to experiment directory
cd exps/<your_exp_id>/

# Check checkpoints were created
ls -lh ckpt/          # Should see epoch*.ckpt and epoch*.torchscript files

# Check final exploitability in logs
grep "Exploitability to leaf" <job_id>_0_log.out | tail -5

# Check final training loss
grep "\[Train\] epoch" <job_id>_0_log.out | tail -5
```

## Verify Poker Variant is Running

**Important**: Make sure you see `Net2Poker` in the logs, not `Net2`!

```bash
# Check model type
grep "Built a model" <job_id>_0_log.out
# Should output: Built a model: Net2Poker(...)

# Check num_actions (should be 15 for poker)
grep "num_actions" <job_id>_0_log.out
```

If you see `Net2` instead of `Net2Poker`, the poker variant isn't being detected!

## Manual Evaluation (Optional)

After training, you can manually evaluate a checkpoint:

```bash
# Build evaluator
make

# Evaluate
build/poker/recursive_eval \
    --net exps/<exp_id>/ckpt/epoch100.torchscript \
    --mdp_depth 4 \
    --subgame_iters 256 \
    --num_repeats 100 \
    --num_threads 8
```

This will output exploitability metrics.
