# Training and Evaluation Guide

## Data Generation vs. Game Logs

### Important: The C++ solver generates **synthetic training data**, not from game logs

The C++ code in `csrc/poker` generates training data by:
1. **Solving the game tree** using CFR/FP algorithms
2. **Computing accurate value estimates** for all possible game states
3. **Generating (state, value) pairs** where:
   - **State** = query encoding (player_id, traverser, action_one_hot, board_cards, discard_choices, street, beliefs)
   - **Value** = expected values for each possible 3-card hand (22,100 values)

This is **different** from using actual game logs because:
- ✅ **More accurate**: Values come from solving, not noisy game outcomes
- ✅ **Complete coverage**: All possible states, not just played ones
- ✅ **Better generalization**: Model learns from full state space

### Your log file format matches our implementation ✓

Your log shows:
- Preflop → Flop (2 cards) → Discard phase → Betting → Turn → River
- 6 board cards by showdown
- Each player discards one card
- Stacks reset to 400 each round

This matches our C++ implementation exactly! The solver will generate data following these same rules.

## Training Process

### 1. Start Training

```bash
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=0  \
    selfplay.threads_per_gpu=8
```

### 2. What Happens During Training

1. **Data Generation** (C++ threads):
   - Solves subgames using CFR/FP
   - Generates (state, value) training pairs
   - Adds to replay buffer

2. **Model Training** (Python):
   - Samples batches from replay buffer
   - Trains neural network to predict values
   - Saves checkpoints every 10 epochs

3. **Evaluation** (every 20 epochs):
   - Computes exploitability
   - Measures how exploitable your strategy is
   - Lower = better (0 = perfect Nash equilibrium)

## Checking Results After Slurm Job

### Quick Check

```bash
# Navigate to experiment directory
cd exps/<your_exp_id>/

# Check what was created
ls -lh ckpt/          # Checkpoints every 10 epochs
ls -lh dumped_data/   # Datasets (if enabled)
```

### Using the Check Script

```bash
# Basic summary
python scripts/check_training_results.py exps/<your_exp_id>/

# With model evaluation
python scripts/check_training_results.py exps/<your_exp_id>/ --eval
```

### What to Look For

#### ✅ Good Signs:
1. **Checkpoints exist**: `ckpt/epoch*.ckpt` and `ckpt/epoch*.torchscript`
2. **Exploitability decreasing**: Check logs for "Exploitability to leaf"
3. **Loss decreasing**: Training loss should decrease over time
4. **Data generation working**: "bps/gen" (batches per second) > 0

#### 📊 Key Metrics (in logs):
- `exploitability_last`: Lower is better (measures strategy quality)
- `loss/train`: Training loss (should decrease)
- `bps/gen`: Data generation speed
- `buffer/size`: Replay buffer size

## Manual Model Evaluation

### Evaluate a Specific Checkpoint

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

### Understanding the Output

The evaluator outputs:
- **Exploitability for full tree**: Solving entire game tree (baseline)
- **Exploitability for recursive solving**: What you actually use (your model)
- **Expected value comparisons**: How your model compares to optimal

### Exploitability Explained

**Exploitability** = How much an opponent could gain by playing optimally against your strategy:
- **0** = Perfect (Nash equilibrium)
- **< 0.1** = Very good for poker variants
- **< 0.01** = Excellent
- **Your goal**: Get as low as possible

## Log Files

### On the Cluster

```bash
# Check job status
squeue -u $USER

# View output
tail -f <job_id>_0_log.out    # Standard output
tail -f <job_id>_0_log.err    # Errors
```

### What to Look For in Logs

1. **Training Progress**:
   ```
   [Train] epoch 10 complete, avg error is 0.0234
   ```

2. **Exploitability** (every 20 epochs):
   ```
   Exploitability to leaf (epoch=20): 0.15
   Exploitability to leaf (epoch=40): 0.12
   Exploitability to leaf (epoch=60): 0.09
   ```
   Should be decreasing!

3. **Data Generation**:
   ```
   time=60 items=50000 per_second=833.33
   ```

4. **Validation**:
   ```
   [Eval] epoch 100 complete, data is valid_snapshot_0100, avg error is 0.021
   ```

## Common Issues

### 1. No Checkpoints Created
- **Cause**: Training crashed early
- **Fix**: Check error logs (`<job_id>_0_log.err`)
- **Common causes**: Out of memory, CUDA errors, config issues

### 2. Exploitability Not Decreasing
- **Cause**: Model not learning
- **Fix**: 
  - Check if training loss is decreasing
  - Verify data generation is working (`bps/gen` > 0)
  - Check learning rate (might be too high/low)

### 3. Low Data Generation Speed
- **Cause**: Bottleneck in data generation
- **Fix**:
  - Use more CPU threads: `selfplay.cpu_gen_threads=60`
  - Use GPUs: `selfplay.cpu_gen_threads=0 selfplay.threads_per_gpu=8`
  - Reduce `subgame_params.num_iters` (faster but less accurate)

### 4. Out of Memory
- **Cause**: Model or buffer too large
- **Fix**:
  - Reduce `replay.capacity` in config
  - Reduce model size (`model.kwargs.n_hidden`)
  - Use smaller batch size (`data.train_batch_size`)

## Using Your Trained Model

Once training is complete, you can:

1. **Load the model in Python**:
   ```python
   import torch
   model = torch.jit.load("ckpt/epoch100.torchscript")
   ```

2. **Use in your poker bot** (`player.py`):
   - Load the TorchScript model
   - Convert game state to query format (match C++ format)
   - Get value predictions
   - Use for decision making

3. **Evaluate against other strategies**:
   - Run tournaments
   - Compare exploitability
   - Test in actual gameplay

## Next Steps

1. ✅ Verify log format matches implementation (use `scripts/verify_log_format.py`)
2. ✅ Start training with your config
3. ✅ Monitor logs for exploitability decreasing
4. ✅ Check results after job completes
5. ✅ Evaluate best checkpoint
6. ✅ Integrate model into your poker bot
