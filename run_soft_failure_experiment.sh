#!/bin/bash

set -euo pipefail

# ============================================================================
# Soft Failure Experiment Script
# - Runs train_kd.py on current workspace
# - Enables soft failover and restart-based recovery path
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmarks/soft_target"
RESULTS_DIR="${PROJECT_ROOT}/results/soft_failure_experiment_$(date +%Y%m%d_%H%M%S)"

# Model paths (relative to BENCHMARK_DIR)
TEACHER_MODEL="./results/base/base-i100-vit-large/model_best.pth.tar"
STUDENT_MODEL="./results/base/base-i100-resnet152/initial_r152.pth.tar"

# Dataset path
IMG_ROOT="/nas-ssd/datasets/imagenet2012/imagenet"

mkdir -p "${RESULTS_DIR}"
echo "Results directory: ${RESULTS_DIR}"

cd "${BENCHMARK_DIR}"

if [[ ! -f "${TEACHER_MODEL}" ]]; then
    echo "Teacher model not found: ${TEACHER_MODEL}"
    exit 1
fi

if [[ ! -f "${STUDENT_MODEL}" ]]; then
    echo "Student model not found: ${STUDENT_MODEL}"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3,4,6}"

echo "Starting soft-failure experiment"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python train_kd.py \
    --img_root="${IMG_ROOT}" \
    --data_name=imagenet100 \
    --num_class=100 \
    --t_name=vit_large \
    --s_name=resnet152 \
    --t_model="${TEACHER_MODEL}" \
    --s_init="${STUDENT_MODEL}" \
    --kd_mode=st \
    --lambda_kd=0.1 \
    --T=4.0 \
    --epochs=1 \
    --batch_size=128 \
    --save_root="${RESULTS_DIR}" \
    --note=soft-failure-tspipe-gpu0346-b128 \
    --tspipe-enable \
    --tspipe-config=./tspipe.yaml \
    --soft-failover-enable \
    --ip=127.0.0.1 \
    --rank=0 \
    --num-nodes=1 \
    2>&1 | tee "${RESULTS_DIR}/run.log"

RUN_DIR="${RESULTS_DIR}/soft-failure-tspipe-gpu0346-b128"

echo "Soft failure experiment completed"
echo "Artifacts: ${RUN_DIR}"

if [[ -f "${RUN_DIR}/failover_restart_config.json" ]]; then
    echo "Found: ${RUN_DIR}/failover_restart_config.json"
fi

if [[ -f "${RUN_DIR}/failover_checkpoint_latest.pth" ]]; then
    echo "Found: ${RUN_DIR}/failover_checkpoint_latest.pth"
fi

echo "Log file: ${RESULTS_DIR}/run.log"
