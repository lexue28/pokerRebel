# Checking Training Results

After your Slurm job completes, here's how to check if your model trained successfully:

## Quick Check

```bash
# Navigate to your experiment directory
cd exps/<your_exp_id>/

# Check what was created
ls -lh ckpt/          # Checkpoints every 10 epochs
ls -lh dumped_data/   # Datasets (if enabled)
```

## Using the Check Script

```bash
# Basic summary
python scripts/check_training_results.py exps/<your_exp_id>/

# With evaluation
python scripts/check_training_results.py exps/<your_exp_id>/ --eval
```

## What to Look For

### ✅ Good Signs:
1. **Checkpoints exist**: `ckpt/epoch*.ckpt` and `ckpt/epoch*.torchscript` files
2. **Exploitability decreasing**: Check logs for "Exploitability to leaf" (computed every 20 epochs)
3. **Loss decreasing**: Training loss should decrease over time
4. **Data generation working**: "bps/gen" (batches per second) should be > 0

### 📊 Key Metrics (in logs):
- `exploitability_last`: Lower is better (measures how exploitable your strategy is)
- `loss/train`: Training loss (should decrease)
- `bps/gen`: Data generation speed
- `buffer/size`: Replay buffer size

## Manual Evaluation

To evaluate a specific checkpoint:

```bash
# Build the evaluator (if not already built)
make

# Evaluate a model
build/poker/recursive_eval \
    --net exps/<exp_id>/ckpt/epoch100.torchscript \
    --mdp_depth 4 \
    --subgame_iters 256 \
    --num_repeats 100 \
    --num_threads 8
```

This will output:
- Exploitability for full tree solving
- Exploitability for recursive solving (what you actually use)
- Expected value comparisons

## Understanding Exploitability

**Exploitability** measures how much an opponent could gain by playing optimally against your strategy:
- **Lower = Better**: 0 means perfect (Nash equilibrium)
- **Typical good values**: < 0.1 for poker variants
- **Your goal**: Get exploitability as low as possible

## Log Files

Check the Slurm output files:
```bash
# On the cluster
tail -f <job_id>_0_log.out    # Standard output
tail -f <job_id>_0_log.err    # Errors
```

Look for:
- `[Train] epoch X complete, avg error is Y` - Training progress
- `Exploitability to leaf (epoch=X): Y` - Strategy quality
- `[Eval] epoch X complete` - Validation metrics

## Common Issues

1. **No checkpoints**: Training might have crashed early - check error logs
2. **Exploitability not decreasing**: Model might not be learning - check loss
3. **Low generation speed**: Data generation might be bottlenecked - check CPU/GPU usage
