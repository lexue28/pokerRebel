# Checkpoint Saving Configuration

## Current Setting
Checkpoints are saved **every 5 epochs**: 0, 5, 10, 15, 20, 25, 30, ...

This is configured in `cfvpy/selfplay.py` line 556:
```python
if self.is_master and epoch % 5 == 0:  # Save checkpoints every 5 epochs
```

## Available Epochs
After training starts, you'll have checkpoints at:
- Epoch 0 (immediately after first epoch)
- Epoch 5
- Epoch 10
- Epoch 15
- Epoch 20
- ... and so on

## To Extract a Checkpoint

### On Remote Server:
```bash
python extract_epoch.py 5   # Extract epoch 5
python extract_epoch.py 10  # Extract epoch 10
```

### Copy to Local:
```bash
scp lexue28@orcd-login.mit.edu:~/rebel/epoch5.torchscript ./
```

## Important: New Job Required
**Yes, you need to run a new training job** for this change to take effect. The current running job was started with the old code (saves every 10 epochs).

To apply the change:
1. Cancel current job (if running): `scancel <job_id>`
2. Make sure the code change is synced to the server
3. Submit a new job: `sbatch run_cpu_batch_balanced.sh`

The new job will save checkpoints every 5 epochs.
