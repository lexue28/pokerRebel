# Pre-Flight Checklist for Slurm Job

## ✅ Before Running Your Command

### 1. Build the C++ Code
```bash
make clean
make
```

**Verify**: Check that `cfvpy/rela*.so` files are created (Python bindings)

### 2. Verify Config File
```bash
# Check poker.yaml exists and has correct game type
cat conf/c02_selfplay/poker.yaml | grep "game:"
# Should output: game: toss_holdem
```

### 3. Test Poker Detection (Optional)
```bash
python -c "import yaml; cfg = yaml.safe_load(open('conf/c02_selfplay/poker.yaml')); print('Game:', cfg['env']['game'])"
# Should output: Game: toss_holdem
```

## ✅ Your Command

```bash
python run.py --adhoc --cfg conf/c02_selfplay/poker.yaml \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=0  \
    selfplay.threads_per_gpu=8
```

## ✅ What to Check in Logs (First 50 lines)

**MUST SEE**:
1. `POKER VARIANT DETECTED: Using Net2Poker model`
2. `Built a model: Net2Poker(...)`
3. `POKER VARIANT: num_actions = 15`

**If you see**:
- `Net2` instead of `Net2Poker` → Config not detected correctly
- `num_actions = <not 15>` → Poker variant not detected
- `AttributeError: 'game'` → Config missing game field

## ✅ Model Size Verification

The model should have:
- **Input size**: 44,226 (query encoding)
- **Output size**: 22,100 (C(52,3) possible hands)
- **Num actions**: 15

Check in logs:
```
model size is <some_number>
```

## ✅ Common Issues

### Issue: "Cannot find key 'game' in cfg"
**Fix**: Make sure `conf/c02_selfplay/poker.yaml` has `env.game: toss_holdem`

### Issue: "ModuleNotFoundError: No module named 'cfvpy.rela'"
**Fix**: Run `make` to build C++ bindings

### Issue: "AttributeError: 'game'"
**Fix**: Config file might be missing the game field - check poker.yaml

### Issue: CUDA device count error
**Fix**: If using `cpu_gen_threads=0`, you need 2+ GPUs. Otherwise set `cpu_gen_threads=8`

## ✅ Success Indicators

After job starts, you should see:
1. ✅ Poker variant detected
2. ✅ Model built successfully
3. ✅ Data generation threads started
4. ✅ Training loop started
5. ✅ Checkpoints being saved (every 10 epochs)

## ✅ After Job Completes

```bash
# Check it worked
grep "POKER VARIANT" <job_id>_0_log.out
grep "Exploitability to leaf" <job_id>_0_log.out | tail -5
ls -lh exps/<exp_id>/ckpt/
```
