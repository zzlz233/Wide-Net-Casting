#!/bin/bash
set -e

# ============================================================================
# 2-GPU smoke test for the group training pipeline on RTX 6000 Ada (48GB).
#
# Goal: verify main_group.py runs end-to-end with 2 branches (llama2 + vicuna)
# on lower-VRAM hardware before scaling up. NOT a full training run.
#
# Hardware assumption: 3x RTX 6000 Ada Gen, using GPUs 0 and 1.
# Dtype: prompter stays fp32 (fp16/bf16 is known to crash mid-training).
# wandb: disabled (no API key configured).
# ============================================================================

# ---- Conda env ----
source /home/xiangq1/anaconda3/etc/profile.d/conda.sh
conda activate CVPR_2026_remiss

# ---- GPU / process config ----
gpu_devices="0,1"
num_gpus=2

# ---- Training hyperparams (same as 4-GPU script, A100-tuned reduced set) ----
lambda_val=150
logprobs=2
opt_type="remiss"

# ---- Output dir ----
timestamp=$(date +%s)
jobname="group2gpu_test_${timestamp}"
output_dir="res_group/${jobname}"
mkdir -p "${output_dir}"

echo "=============================================="
echo "Starting 2-GPU SMOKE TEST: ${jobname}"
echo "GPUs: [${gpu_devices}]"
echo "Branches: llama2_chat (rank0) + vicuna_chat (rank1)"
echo "Params: lambda=${lambda_val}, logprobs=${logprobs}, opt_type=${opt_type}"
echo "Output: ${output_dir}"
echo "=============================================="

# Pick a free rendezvous port so we don't collide with other torchrun jobs.
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))

CUDA_VISIBLE_DEVICES=${gpu_devices} torchrun \
    --standalone \
    --nproc_per_node=${num_gpus} \
    --master_port=${MASTER_PORT} \
    main_group.py \
    --config-name=train_group_2gpu \
    verbose=true \
    train.epochs=1 \
    train.batch_size=1 \
    train.eval_every=999 \
    train.do_initial_eval=false \
    train.q_params.num_beams=2 \
    train.q_params.num_chunks=2 \
    train.q_params.top_k=12 \
    train.q_params.lambda_val=${lambda_val} \
    train.q_params.selected_logprobs=${logprobs} \
    train.opt_type=${opt_type} \
    wandb_params.enable_wandb=false \
    wandb_params.id=${jobname} \
    output_dir=${output_dir} \
    2>&1 | tee "${output_dir}/train.log"

echo "=============================================="
echo "Smoke test finished. Inspect: ${output_dir}/train.log"
echo "Per-branch logs: ${output_dir}/llama2_rank0/  ${output_dir}/vicuna_rank1/"
echo "=============================================="
