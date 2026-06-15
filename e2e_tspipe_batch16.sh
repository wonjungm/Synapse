#!/bin/bash
set -euo pipefail

unset FAILOVER_INJECT_SCENARIO

# Keep the same launcher/training path as the failover run.
export FAILOVER_TEST_FAST_GATES="${FAILOVER_TEST_FAST_GATES:-0}"
export FAILOVER_SLOWDOWN_THRESHOLD_SEC="${FAILOVER_SLOWDOWN_THRESHOLD_SEC:-1000000000.0}"
export DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS:-0,1,2,3}"

export RUN_NOTE_BASE="${RUN_NOTE_BASE:-e2e_tspipe_batch16}"
export RUN_NOTE="${RUN_NOTE:-${RUN_NOTE_BASE}_$(date +%Y%m%d_%H%M%S)}"
export SLOWDOWN_GPU="${SLOWDOWN_GPU:-0}"
export MAX_STEPS_PER_EPOCH=0
export TOTAL_STEPS="${TOTAL_STEPS:-7912}"
EFFECTIVE_TOTAL_STEPS="${TOTAL_STEPS}"

SLOWDOWN_START_DEFAULT=$(( (EFFECTIVE_TOTAL_STEPS * 5 + 99) / 100 ))
if [[ "${SLOWDOWN_START_DEFAULT}" -lt 1 ]]; then
  SLOWDOWN_START_DEFAULT=1
fi
SLOWDOWN_END_DEFAULT=$(( EFFECTIVE_TOTAL_STEPS / 2 ))
if [[ "${SLOWDOWN_END_DEFAULT}" -lt "${SLOWDOWN_START_DEFAULT}" ]]; then
  SLOWDOWN_END_DEFAULT="${SLOWDOWN_START_DEFAULT}"
fi

export SLOWDOWN_MODE="${SLOWDOWN_MODE:-fixed}"
export SLOWDOWN_FIXED_MS="${SLOWDOWN_FIXED_MS:-500}"
export SLOWDOWN_START="${SLOWDOWN_START:-${SLOWDOWN_START_DEFAULT}}"
export SLOWDOWN_END="${SLOWDOWN_END:-${SLOWDOWN_END_DEFAULT}}"
export TSPIPE_CONFIG="${TSPIPE_CONFIG:-benchmarks/soft_target/tspipe.yaml}"

bash ./run_e2e_failover.sh \
  --img_root=/workspace/datasets/imagenet \
  --data_name=imagenet100 \
  --t_name=vit_large \
  --s_name=resnet152 \
  --kd_mode=st \
  --lambda_kd=0.1 \
  --t_model=/workspace/Synapse/Synapse/benchmarks/soft_target/results/base/base-i100-vit-large/model_best.pth.tar \
  --s_init=/workspace/Synapse/Synapse/benchmarks/soft_target/results/base/base-i100-resnet152/initial_r152.pth.tar \
  --batch_size=16 \
  --num_class=100 \
  --epochs=1 \
  --max-steps-per-epoch="${MAX_STEPS_PER_EPOCH}" \
  --tspipe-enable \
  --tspipe-config="${TSPIPE_CONFIG}" \
  --inject-slowdown-gpu="${SLOWDOWN_GPU}" \
  --slowdown-task-scope=compute \
  --slowdown-mode="${SLOWDOWN_MODE}" \
  --slowdown-fixed-ms="${SLOWDOWN_FIXED_MS}" \
  --slowdown-start="${SLOWDOWN_START}" \
  --slowdown-end="${SLOWDOWN_END}"