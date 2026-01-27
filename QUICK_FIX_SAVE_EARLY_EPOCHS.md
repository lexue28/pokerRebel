# Quick Fix: Save Early Epochs (Epoch 0-10)

To get epoch 4, modify the checkpoint saving logic to save every epoch for the first 10 epochs.

## Edit `cfvpy/selfplay.py`

Find line 556:
```python
if self.is_master and epoch % 10 == 0:
```

Change it to:
```python
if self.is_master and (epoch % 10 == 0 or epoch < 10):
```

This will save checkpoints at epochs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40...

## After Making This Change

1. Rebuild if needed (usually not needed for Python changes)
2. Restart training
3. Wait for epoch 4 to complete
4. Run: `python extract_epoch.py 4`

## Alternative: Save Every Epoch (Warning: Uses More Disk Space)

If you want to save every epoch:
```python
if self.is_master:  # Save every epoch
```

This uses more disk space but gives you maximum flexibility.
