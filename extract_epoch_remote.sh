#!/bin/bash
# Script to extract epoch checkpoint on remote server
# Run this on the Engaging portal after training starts
# Usage: ./extract_epoch_remote.sh [epoch_number]

EPOCH=${1:-4}
BASE_DIR="${HOME}/rebel"
EXPS_DIR="${BASE_DIR}/exps"

echo "Looking for epoch ${EPOCH} checkpoint in ${EXPS_DIR}..."

# Find most recent checkpoint directory
CKPT_DIR=$(find "${EXPS_DIR}" -type d -name "ckpt" | sort -r | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint directory found in ${EXPS_DIR}"
    echo "Make sure training has started and checkpoints are being saved."
    exit 1
fi

echo "Found checkpoint directory: ${CKPT_DIR}"

# Try to find exact epoch
EPOCH_FILE="${CKPT_DIR}/epoch${EPOCH}.torchscript"

if [ -f "$EPOCH_FILE" ]; then
    echo "Found exact match: ${EPOCH_FILE}"
    TARGET="${BASE_DIR}/epoch${EPOCH}.torchscript"
    cp "$EPOCH_FILE" "$TARGET"
    echo "✓ Copied to: ${TARGET}"
    exit 0
fi

# Find closest epoch
echo "Epoch ${EPOCH} not found. Looking for closest available epoch..."
AVAILABLE=$(ls -1 "${CKPT_DIR}"/epoch*.torchscript 2>/dev/null | sed 's/.*epoch\([0-9]*\)\.torchscript/\1/' | sort -n)

if [ -z "$AVAILABLE" ]; then
    echo "ERROR: No epoch files found in ${CKPT_DIR}"
    exit 1
fi

# Find closest epoch <= target
CLOSEST=""
for e in $AVAILABLE; do
    if [ "$e" -le "$EPOCH" ]; then
        CLOSEST="$e"
    else
        break
    fi
done

# If no epoch <= target, use smallest
if [ -z "$CLOSEST" ]; then
    CLOSEST=$(echo "$AVAILABLE" | head -1)
    echo "WARNING: No epoch <= ${EPOCH} found. Using smallest available: epoch ${CLOSEST}"
else
    echo "Found closest epoch <= ${EPOCH}: epoch ${CLOSEST}"
fi

EPOCH_FILE="${CKPT_DIR}/epoch${CLOSEST}.torchscript"
TARGET="${BASE_DIR}/epoch${CLOSEST}.torchscript"

cp "$EPOCH_FILE" "$TARGET"
echo "✓ Copied epoch ${CLOSEST} to: ${TARGET}"
echo ""
echo "To copy to local machine, run from your local terminal:"
echo "  scp lexue28@orcd-login.mit.edu:${TARGET} ./epoch${CLOSEST}.torchscript"
