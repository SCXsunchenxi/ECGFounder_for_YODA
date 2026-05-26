#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ECGFounder — single-lead inference on legacy Xy matrix H5 files
#
# Data format:  legacy Xy matrix (shape: n_channels × n_samples)
# ECG channel:  ch13 (0-indexed), sampling rate 200 Hz
# Model:        1-lead ECGFounder (in_channels=1)
# Output:       results/<h5_stem>.csv  (one CSV per H5 file)
#   columns: file, window_start_sec, window_end_sec, <150 class probabilities>
#
# If the checkpoint is missing, download it from HuggingFace:
#   pip install huggingface_hub
#   python -c "from huggingface_hub import hf_hub_download; \
#     hf_hub_download('PKUDigitalHealth/ECGFounder', '1_lead_ECGFounder.pth', \
#     local_dir='./checkpoint')"
# ─────────────────────────────────────────────────────────────────────────────

H5_DIR="/data/sleep/ECGFounder/S0001_500_age_stratified_data/"
CKPT="./checkpoint/1_lead_ECGFounder.pth"
TASKS="./tasks.txt"
OUT="/data/sleep/ECGFounder/S0001_500_age_stratified_results/"           # output directory; one <stem>.csv per H5 file
BATCH_SIZE=256

# H5 files created by cohort_h5_conversion_reference.py store ECG under
# /signals/ecg  (grouped format).  Use --leads to activate that mode.
LEADS="ecg"

WINDOW_SEC=10    # sliding-window length in seconds
STRIDE_SEC=10    # stride between windows (= no overlap)

# yes = reprocess all files; no = skip files whose output CSV already exists
REWRITE="yes"

# ── Checkpoint check ──────────────────────────────────────────────────────
if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    echo "        Download it with:"
    echo "          pip install huggingface_hub"
    echo "          python -c \"from huggingface_hub import hf_hub_download; \\"
    echo "            hf_hub_download('PKUDigitalHealth/ECGFounder', '1_lead_ECGFounder.pth', \\"
    echo "            local_dir='./checkpoint')\""
    exit 1
fi

mkdir -p "$OUT"

REWRITE_FLAG=""
[ "$REWRITE" = "yes" ] && REWRITE_FLAG="--rewrite"

python infer_h5.py \
    --h5_dir     "$H5_DIR" \
    --ckpt       "$CKPT" \
    --tasks      "$TASKS" \
    --out        "$OUT" \
    --batch_size "$BATCH_SIZE" \
    --leads      "$LEADS" \
    --window_sec "$WINDOW_SEC" \
    --stride_sec "$STRIDE_SEC" \
    $REWRITE_FLAG
