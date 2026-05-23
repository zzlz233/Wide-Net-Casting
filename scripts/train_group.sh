#!/bin/bash
set -e

# GPU selection
gpu_devices="0,1,2,3"
num_gpus=4

lambda_val=150
logprobs=2
opt_type="remiss"

# Job name and output dir
timestamp=$(date +%s)
jobname="group_${timestamp}"
output_dir="res_group/${jobname}"
mkdir -p "${output_dir}"

echo "Starting ${num_gpus}GPU training on GPUs [${gpu_devices}]: ${jobname}"
echo "Parameters: lambda=${lambda_val}, logprobs=${logprobs}, opt_type=${opt_type}"

# Set CUDA_VISIBLE_DEVICES and launch distributed training
CUDA_VISIBLE_DEVICES=${gpu_devices} torchrun \
    --standalone \
    --nproc_per_node=${num_gpus} \
    main_group.py \
    --config-name=train_group \
    verbose=true \
    train.batch_size=2 \
    train.q_params.num_beams=2 \
    train.q_params.num_chunks=2 \
    train.q_params.top_k=12 \
    train.q_params.lambda_val=${lambda_val} \
    train.q_params.selected_logprobs=${logprobs} \
    wandb_params.id=${jobname} \
    train.eval_every=1 \
    output_dir=${output_dir} \
    train.opt_type=${opt_type} \
    2>&1 | tee "${output_dir}/train.log"

echo "Training completed: ${output_dir}"