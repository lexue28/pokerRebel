# Extract Epoch Checkpoint Scripts

## Problem
ReBeL saves checkpoints every 10 epochs (0, 10, 20, 30...), so epoch 4 won't exist by default.

## Solutions

### Option 1: Use Available Checkpoint (Epoch 0)
Epoch 0 should exist. Extract it:
```bash
python extract_epoch.py 0
```

### Option 2: Modify Training to Save Every Epoch
Edit `cfvpy/selfplay.py` line 556:
```python
# Change from:
if self.is_master and epoch % 10 == 0:

# To:
if self.is_master and (epoch % 10 == 0 or epoch < 10):  # Save every epoch for first 10
```

Or save every epoch:
```python
if self.is_master:  # Save every epoch
```

### Option 3: Use Closest Available Epoch
The script will automatically find the closest available epoch:
```bash
python extract_epoch.py 4  # Will use epoch 0 if 4 doesn't exist
```

## Usage

### Local (after copying files from server)
```bash
# List available epochs
python extract_epoch.py --list

# Extract epoch 4 (or closest)
python extract_epoch.py 4

# Extract to specific location
python extract_epoch.py 4 ../clan-pokerbot/epoch4.torchscript
```

### Remote (on Engaging portal)
```bash
# Make script executable
chmod +x extract_epoch_remote.sh

# Extract epoch 4 (or closest)
./extract_epoch_remote.sh 4

# Then copy to local machine
scp lexue28@orcd-login.mit.edu:~/rebel/epoch4.torchscript ./
```

## Quick Fix: Use Epoch 0 Now
If you need something immediately:
```bash
# On remote server
find ~/rebel/exps -name "epoch0.torchscript" -exec cp {} ~/rebel/epoch0.torchscript \;

# Copy to local
scp lexue28@orcd-login.mit.edu:~/rebel/epoch0.torchscript ./epoch0.torchscript

# Update player.py to use epoch0.torchscript
```
