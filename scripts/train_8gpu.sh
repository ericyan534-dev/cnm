#!/bin/bash
#
# CNM-BERT 8xGPU Training Script
#
# This script launches distributed pretraining on 8 GPUs using torchrun
# within a tmux session named "train".
#
# Usage:
#   ./scripts/train_8gpu.sh
#
# Prerequisites:
#   - conda environment 'cnm' must be set up
#   - IDS data must be prepared (run scripts/download_ids.py && scripts/prepare_ids.py)
#   - Training corpus must be available at data/corpus/
#

set -e

# Configuration
SESSION_NAME="train"
NUM_GPUS=8
CONDA_ENV="cnm"

# Training parameters
TRAIN_FILE="data/corpus"
CNM_VOCAB_PATH="data/ids/cnm_vocab.json"
OUTPUT_DIR="outputs/pretrain-8gpu-$(date +%Y%m%d-%H%M%S)"

# Hyperparameters
NUM_EPOCHS=3
BATCH_SIZE_PER_GPU=32
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-4
WARMUP_RATIO=0.1
MAX_SEQ_LENGTH=512
MLM_PROBABILITY=0.15
AUX_LOSS_WEIGHT=0.1

# Logging
LOGGING_STEPS=100
SAVE_STEPS=1000
EVAL_STEPS=1000
SAVE_TOTAL_LIMIT=3
REPORT_TO="wandb"
RUN_NAME="cnm-bert-pretrain-8gpu"

# Build the training command
TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS} scripts/pretrain.py \
    --train_file ${TRAIN_FILE} \
    --cnm_vocab_path ${CNM_VOCAB_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --pretrained_bert bert-base-chinese \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay 0.01 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --mlm_probability ${MLM_PROBABILITY} \
    --aux_loss_weight ${AUX_LOSS_WEIGHT} \
    --wwm \
    --bf16 \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --report_to ${REPORT_TO} \
    --run_name ${RUN_NAME} \
    --seed 42"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed. Please install tmux first."
    exit 1
fi

# Check if session already exists
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "WARNING: tmux session '${SESSION_NAME}' already exists."
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t ${SESSION_NAME}"
    echo "  2. Kill and recreate: tmux kill-session -t ${SESSION_NAME}"
    echo ""
    read -p "Kill existing session and start new training? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t ${SESSION_NAME}
    else
        echo "Aborting. Use 'tmux attach -t ${SESSION_NAME}' to attach to existing session."
        exit 0
    fi
fi

echo "=========================================="
echo "CNM-BERT 8xGPU Distributed Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Effective batch size: $((NUM_GPUS * BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Output: ${OUTPUT_DIR}"
echo ""
echo "Starting training in tmux session '${SESSION_NAME}'..."
echo ""

# Create tmux session and run training
tmux new-session -d -s ${SESSION_NAME} \
    "source ~/.bashrc; \
     conda activate ${CONDA_ENV}; \
     cd $(pwd); \
     echo '=== CNM-BERT Training Started ==='; \
     echo 'Output directory: ${OUTPUT_DIR}'; \
     echo ''; \
     ${TRAIN_CMD}; \
     echo ''; \
     echo '=== Training Complete ==='; \
     echo 'Press Enter to close...'; \
     read"

echo "Training started in tmux session '${SESSION_NAME}'"
echo ""
echo "To monitor training:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "To detach from session:"
echo "  Press Ctrl+B, then D"
echo ""
echo "To kill training:"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo ""
