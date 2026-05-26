#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ECGFounder — inference on AFib case/control dataset
#
# Input layout (produced by download_from_aws.py):
#   afib_case_control_200_data/afib_positive/<age_group>/*.h5
#   afib_case_control_200_data/afib_negative/<age_group>/*.h5
#
# Output layout (mirrors input structure):
#   afib_case_control_200_results/afib_positive/<age_group>/<stem>.csv
#   afib_case_control_200_results/afib_negative/<age_group>/<stem>.csv
#
# If the checkpoint is missing, download it from HuggingFace:
#   pip install huggingface_hub
#   python -c "from huggingface_hub import hf_hub_download; \
#     hf_hub_download('PKUDigitalHealth/ECGFounder', '1_lead_ECGFounder.pth', \
#     local_dir='./checkpoint')"
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT="/data/YODA/ECGFounder/afib_case_control_200_data"
OUT_ROOT="/data/YODA/ECGFounder/afib_case_control_200_results"
CKPT="./checkpoint/1_lead_ECGFounder.pth"
TASKS="./tasks.txt"
BATCH_SIZE=256
LEADS="ecg"
WINDOW_SEC=10
STRIDE_SEC=10

# yes = reprocess all files; no = skip files whose output CSV already exists
REWRITE="no"

# ── Checkpoint check ──────────────────────────────────────────────────────────
if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    echo "        Download it with:"
    echo "          pip install huggingface_hub"
    echo "          python -c \"from huggingface_hub import hf_hub_download; \\"
    echo "            hf_hub_download('PKUDigitalHealth/ECGFounder', '1_lead_ECGFounder.pth', \\"
    echo "            local_dir='./checkpoint')\""
    exit 1
fi

REWRITE_FLAG=""
[ "$REWRITE" = "yes" ] && REWRITE_FLAG="--rewrite"

# ── Loop over afib_positive / afib_negative → age_group subdirectories ───────
for LABEL_DIR in "$DATA_ROOT"/afib_positive "$DATA_ROOT"/afib_negative; do
    [ -d "$LABEL_DIR" ] || continue
    LABEL_NAME=$(basename "$LABEL_DIR")   # afib_positive or afib_negative

    for AGE_DIR in "$LABEL_DIR"/*/; do
        [ -d "$AGE_DIR" ] || continue
        AGE_NAME=$(basename "$AGE_DIR")   # e.g. 50-60

        H5_DIR="$AGE_DIR"
        OUT="$OUT_ROOT/$LABEL_NAME/$AGE_NAME"
        sudo mkdir -p "$OUT"

        echo "── $LABEL_NAME / $AGE_NAME ──────────────────────────────"
        sudo $(which python) infer_h5.py \
            --h5_dir     "$H5_DIR" \
            --ckpt       "$CKPT" \
            --tasks      "$TASKS" \
            --out        "$OUT" \
            --batch_size "$BATCH_SIZE" \
            --leads      "$LEADS" \
            --window_sec "$WINDOW_SEC" \
            --stride_sec "$STRIDE_SEC" \
            $REWRITE_FLAG
    done
done

echo "All done. Results saved to $OUT_ROOT"
