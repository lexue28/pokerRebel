#!/usr/bin/env python3
"""
Extract a specific epoch checkpoint from ReBeL training output.
Finds the checkpoint directory and copies the .torchscript file.

Usage:
    python extract_epoch.py [epoch_number] [target_path]
    
Examples:
    python extract_epoch.py 4                    # Extract epoch 4 to current dir
    python extract_epoch.py 4 epoch4.torchscript # Extract to specific file
    python extract_epoch.py 4 ../clan-pokerbot/epoch4.torchscript  # Extract to pokerbot dir
"""

import argparse
import pathlib
import shutil
import sys
from pathlib import Path


def find_checkpoint_dirs(base_dir="exps"):
    """Find all checkpoint directories in exps/."""
    base = Path(base_dir)
    if not base.exists():
        return []
    
    checkpoint_dirs = []
    # Look for ckpt directories in exps/adhoc/.../.../ckpt
    for exp_dir in base.rglob("ckpt"):
        if exp_dir.is_dir():
            checkpoint_dirs.append(exp_dir)
    
    return sorted(checkpoint_dirs, key=lambda x: x.stat().st_mtime, reverse=True)


def find_epoch_file(ckpt_dir, epoch_num):
    """Find epoch file, trying exact match first, then closest."""
    ckpt_path = Path(ckpt_dir)
    
    # Try exact match first
    exact_file = ckpt_path / f"epoch{epoch_num}.torchscript"
    if exact_file.exists():
        return exact_file, epoch_num
    
    # Find all epoch files
    epoch_files = sorted(ckpt_path.glob("epoch*.torchscript"))
    if not epoch_files:
        return None, None
    
    # Extract epoch numbers
    epochs = []
    for f in epoch_files:
        try:
            # Extract number from "epoch4.torchscript"
            num_str = f.stem.replace("epoch", "")
            epoch = int(num_str)
            epochs.append((epoch, f))
        except ValueError:
            continue
    
    if not epochs:
        return None, None
    
    epochs.sort(key=lambda x: x[0])
    
    # Find closest epoch <= target (prefer exact or lower)
    best_epoch = None
    best_file = None
    for epoch, f in epochs:
        if epoch <= epoch_num:
            best_epoch = epoch
            best_file = f
        else:
            break
    
    # If no epoch <= target, use smallest available (but warn)
    if best_file is None:
        best_epoch, best_file = epochs[0]
        print(f"WARNING: Epoch {epoch_num} not found. Using closest available: epoch {best_epoch}")
    elif best_epoch < epoch_num:
        print(f"WARNING: Epoch {epoch_num} not found. Using closest available: epoch {best_epoch}")
    
    return best_file, best_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Extract epoch checkpoint from ReBeL training"
    )
    parser.add_argument(
        "epoch",
        type=int,
        nargs="?",
        default=4,
        help="Epoch number to extract (default: 4). Checkpoints are saved every 5 epochs (0, 5, 10, 15, 20, ...)",
    )
    parser.add_argument(
        "target",
        type=str,
        nargs="?",
        default=None,
        help="Target file path (default: epoch{num}.torchscript in current dir)",
    )
    parser.add_argument(
        "--exps-dir",
        type=str,
        default="exps",
        help="Base directory for experiments (default: exps)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available epochs in checkpoint directories",
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("Searching for checkpoint directories...")
        ckpt_dirs = find_checkpoint_dirs(args.exps_dir)
        if not ckpt_dirs:
            print(f"No checkpoint directories found in {args.exps_dir}/")
            return 1
        
        print(f"\nFound {len(ckpt_dirs)} checkpoint directory(ies):\n")
        for i, ckpt_dir in enumerate(ckpt_dirs, 1):
            print(f"{i}. {ckpt_dir}")
            epoch_files = sorted(ckpt_dir.glob("epoch*.torchscript"))
            if epoch_files:
                epochs = []
                for f in epoch_files:
                    try:
                        num_str = f.stem.replace("epoch", "")
                        epoch = int(num_str)
                        epochs.append(epoch)
                    except ValueError:
                        continue
                epochs.sort()
                print(f"   Available epochs: {epochs}")
                if epochs:
                    print(f"   Latest: epoch{epochs[-1]}")
            else:
                print("   No epoch files found")
            print()
        return 0
    
    # Extract mode
    print(f"Looking for epoch {args.epoch} checkpoint...")
    ckpt_dirs = find_checkpoint_dirs(args.exps_dir)
    
    if not ckpt_dirs:
        print(f"ERROR: No checkpoint directories found in {args.exps_dir}/")
        print("Make sure you've run training and checkpoints exist.")
        return 1
    
    # Try each checkpoint directory (newest first)
    found_file = None
    found_epoch = None
    found_dir = None
    
    for ckpt_dir in ckpt_dirs:
        file, epoch = find_epoch_file(ckpt_dir, args.epoch)
        if file:
            found_file = file
            found_epoch = epoch
            found_dir = ckpt_dir
            break
    
    if not found_file:
        print(f"ERROR: Could not find epoch {args.epoch} (or any epoch) in checkpoint directories")
        print(f"\nSearched in:")
        for ckpt_dir in ckpt_dirs:
            print(f"  - {ckpt_dir}")
        print("\nUse --list to see available epochs")
        return 1
    
    # Determine target path
    if args.target:
        target_path = Path(args.target)
    else:
        target_path = Path.cwd() / f"epoch{found_epoch}.torchscript"
    
    # Copy file
    print(f"Found: {found_file} (epoch {found_epoch})")
    print(f"Copying to: {target_path}")
    
    try:
        shutil.copy2(found_file, target_path)
        print(f"✓ Successfully copied epoch {found_epoch} to {target_path}")
        
        # Also check if .ckpt exists and offer to copy it
        ckpt_file = found_dir / f"epoch{found_epoch}.ckpt"
        if ckpt_file.exists():
            print(f"\nNote: Also found {ckpt_file}")
            print("   (This is the PyTorch checkpoint, .torchscript is what you need for player.py)")
        
        return 0
    except Exception as e:
        print(f"ERROR: Failed to copy file: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
