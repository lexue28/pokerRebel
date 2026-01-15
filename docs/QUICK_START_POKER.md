# Quick Start: Training Poker Variant

## Command to Run

```bash
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=0  \
    selfplay.threads_per_gpu=8
```

## Verify Poker Variant is Running

**Check the logs immediately after starting** - you should see:

```
POKER VARIANT DETECTED: Using Net2Poker model
Built a model: Net2Poker(...)
POKER VARIANT: num_actions = 15
```

If you see `Net2` instead of `Net2Poker`, or `num_actions` is not 15, something is wrong!

## What the Logs Show (Original Method)

Just check the Slurm output files - no special script needed:

```bash
tail -f <job_id>_0_log.out    # Standard output
tail -f <job_id>_0_log.err    # Errors
```

### Key Log Messages

1. **Startup** (should show poker):
   ```
   POKER VARIANT DETECTED: Using Net2Poker model
   Built a model: Net2Poker(...)
   POKER VARIANT: num_actions = 15
   ```

2. **Training Progress** (every epoch):
   ```
   [Train] epoch 0 complete, avg error is 0.0234
   [Train] epoch 10 complete, avg error is 0.0198
   ```
   - Error should decrease

3. **Exploitability** (every 20 epochs):
   ```
   Exploitability to leaf (epoch=20): 0.15
   Exploitability to leaf (epoch=40): 0.12
   ```
   - Should decrease over time
   - Lower = better (0 = perfect)

4. **Checkpoints** (every 10 epochs):
   - Saved to `ckpt/epoch*.ckpt` and `ckpt/epoch*.torchscript`
   - Not logged, just check the directory

5. **Final Metrics** (end of each epoch):
   ```
   Metrics: {'loss/train': 0.016, 'exploitability_last': 0.09, 'bps/gen': 0.5, ...}
   ```

## After Job Completes

```bash
# Check checkpoints
cd exps/<your_exp_id>/
ls -lh ckpt/

# Check final exploitability
grep "Exploitability to leaf" <job_id>_0_log.out | tail -5

# Verify poker was used
grep "POKER VARIANT" <job_id>_0_log.out
```

## Differences from Liar's Dice

- **Model**: `Net2Poker` instead of `Net2`
- **Actions**: 15 instead of dice-specific calculation
- **Input size**: 44,232 (poker query format)
- **Output size**: 22,100 (C(52,3) possible hands)
- **No dice parameters**: `num_dice` and `num_faces` not needed

## Troubleshooting

**If you see dice model being used:**
- Check config has `env.game: toss_holdem`
- Verify config file path is correct
- Check logs for "POKER VARIANT DETECTED" message

**If exploitability not decreasing:**
- Check training loss is decreasing
- Verify data generation is working (`bps/gen` > 0)
- Check learning rate
