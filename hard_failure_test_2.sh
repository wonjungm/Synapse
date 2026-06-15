#!/usr/bin/env bash
set -euo pipefail

# Paper-oriented hard-failure run.
# Defaults are chosen to make the post-recovery slope easier to see:
# - batch_size=64: fewer stages can make the K-1 repartition benefit more visible
# - fail_after_batches=100: past warmup, but with enough work left to amortize recovery
# - max_steps_per_epoch=360: gives the recovered run room to separate from the baseline
#
# If the recovered line still looks too parallel:
# 1. increase MAX_STEPS to 480
# 2. then try BATCH_SIZE=32
# 3. keep FAIL_AFTER_BATCHES aligned with HEALTHY_CKPT_INTERVAL

export E2E_MASTER_IP="${E2E_MASTER_IP:-127.0.0.1}"
export DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS:-0,2,5,6}"
export MAX_RESTARTS="${MAX_RESTARTS:-3}"
unset FAILOVER_INJECT_SCENARIO
unset FAILOVER_TEST_FAST_GATES
unset FAILOVER_SLOWDOWN_THRESHOLD_SEC

export RUN_NOTE="${RUN_NOTE:-paper_hard_test_2}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-360}"
TARGET_FAIL_GPU="${TARGET_FAIL_GPU:-1}"
FAIL_AFTER_BATCHES="${FAIL_AFTER_BATCHES:-100}"
HEALTHY_CKPT_INTERVAL="${HEALTHY_CKPT_INTERVAL:-20}"

echo "[hard_failure_test_2] Using benchmarks/soft_target/tspipe.yaml as-is"
echo "[hard_failure_test_2] batch_size=${BATCH_SIZE} fail_after=${FAIL_AFTER_BATCHES} max_steps=${MAX_STEPS}"

bash ./run_e2e_failover.sh \
  --img_root=/nas-ssd/datasets/imagenet2012/imagenet \
  --data_name=imagenet100 \
  --t_name=vit_large \
  --s_name=resnet152 \
  --kd_mode=st \
  --lambda_kd=0.1 \
  --t_model=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-vit-large/model_best.pth.tar \
  --s_init=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-resnet152/initial_r152.pth.tar \
  --batch_size="${BATCH_SIZE}" \
  --num_class=100 \
  --epochs="${EPOCHS}" \
  --max-steps-per-epoch="${MAX_STEPS}" \
  --tspipe-enable \
  --tspipe-config=benchmarks/soft_target/tspipe.yaml \
  --failover-enable \
  --target-fail-gpu="${TARGET_FAIL_GPU}" \
  --fail-after-batches="${FAIL_AFTER_BATCHES}" \
  --health-check-interval=1 \
  --healthy-checkpoint-interval="${HEALTHY_CKPT_INTERVAL}"
