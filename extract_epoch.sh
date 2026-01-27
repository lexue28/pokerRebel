#!/bin/bash
# Quick script to extract epoch 4 checkpoint
# Usage: ./extract_epoch.sh [epoch_number] [target_path]

EPOCH=${1:-4}
TARGET=${2:-"../clan-pokerbot/epoch${EPOCH}.torchscript"}

echo "Extracting epoch ${EPOCH} checkpoint..."
python extract_epoch.py "${EPOCH}" "${TARGET}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Done! Update player.py to use:"
    echo "  _MODEL_PATH = Path(__file__).resolve().parent.parent / \"epoch${EPOCH}.torchscript\""
else
    echo ""
    echo "✗ Failed. Try:"
    echo "  python extract_epoch.py --list  # See available epochs"
    exit 1
fi
