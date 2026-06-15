#!/usr/bin/env bash
set -euo pipefail

# End-to-end failover launcher:
# - exit 42: failover-triggered restart required
# - exit 0: normal completion
# - others : real failure
FAILOVER_EXIT_CODE=42

# Configure run identity (used by train_kd.py: args.save_root/args.note)
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-./results}"
RUN_NOTE="${RUN_NOTE:-e2e_failover}"
RUN_DIR="${BASE_SAVE_ROOT}/${RUN_NOTE}"
E2E_SUMMARY_PATH="${RUN_DIR}/e2e_summary.log"
SOFT_RESTART_CONFIG_PATH="${RUN_DIR}/failover_restart_config.json"
HARD_RESTART_CONFIG_PATH="${RUN_DIR}/emergency_restart_config.json"
LEGACY_RESTART_CONFIG_PATH="${RUN_DIR}/restart_config.json"
E2E_MASTER_IP="${E2E_MASTER_IP:-127.0.0.1}"

# Default GPU set for initial boot (before any restart config exists).
DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS:-0,1,2,3}"

# Optional restart limit to avoid infinite loops during debugging.
MAX_RESTARTS="${MAX_RESTARTS:-0}"  # 0 means unlimited

# RESTART_COUNT must be initialized outside the loop to persist across restarts
RESTART_COUNT=0

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
Usage:
  ./run_e2e_failover.sh <train_kd.py args...>

Example:
  ./run_e2e_failover.sh \
    --data_name imagenet100 \
    --t_name vit_large \
    --s_name resnet152 \
    --kd_mode st \
    --s_init ./results/base/base-i100-resnet152/initial_r152.pth.tar \
    --t_model ./results/base/base-i100-vit-large/model_best.pth.tar

Environment variables:
  BASE_SAVE_ROOT           Base save directory (default: ./results)
  RUN_NOTE                 Run note subdir (default: e2e_failover)
  DEFAULT_VISIBLE_GPUS     Initial visible GPUs before restart config exists (default: 0,3,4,6)
  MAX_RESTARTS             Max failover restarts; 0 for unlimited (default: 0)
  FAILOVER_INJECT_SCENARIO Inject synthetic slowdown (e.g., KEEP_REPLAN_DEGRADE)
  FAILOVER_TEST_FAST_GATES Enable fast gate detection for quick testing
USAGE
  exit 2
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tspipe
fi

mkdir -p "${RUN_DIR}"

echo "[E2E] BASE_SAVE_ROOT=${BASE_SAVE_ROOT} RUN_NOTE=${RUN_NOTE} RUN_DIR=${RUN_DIR}"
echo "[E2E] Starting failover loop..."

# Measure end-to-end wall-clock time across all restarts
E2E_START_TIME=$(date +%s)
E2E_START_TS=$(date '+%Y-%m-%d %H:%M:%S')
E2E_SUMMARY_WRITTEN=0
E2E_FINAL_STATUS="RUNNING"

write_e2e_summary() {
  local exit_code="${1:-0}"
  local end_time elapsed_sec elapsed_min elapsed_rem end_ts

  if [[ "${E2E_SUMMARY_WRITTEN}" -eq 1 ]]; then
    return
  fi

  end_time=$(date +%s)
  elapsed_sec=$((end_time - E2E_START_TIME))
  if [[ ${elapsed_sec} -lt 0 ]]; then
    elapsed_sec=0
  fi
  elapsed_min=$((elapsed_sec / 60))
  elapsed_rem=$((elapsed_sec % 60))
  end_ts=$(date '+%Y-%m-%d %H:%M:%S')

  {
    echo "[E2E] Run note: ${RUN_NOTE}"
    echo "[E2E] Run dir: ${RUN_DIR}"
    echo "[E2E] Status: ${E2E_FINAL_STATUS}"
    echo "[E2E] Exit code: ${exit_code}"
    echo "[E2E] Restart count: ${RESTART_COUNT}"
    echo "[E2E] Start time: ${E2E_START_TS}"
    echo "[E2E] End time:   ${end_ts}"
    echo "[E2E] Total wall-clock time: ${elapsed_sec}s (${elapsed_min}m ${elapsed_rem}s)"
  } | tee "${E2E_SUMMARY_PATH}"

  E2E_SUMMARY_WRITTEN=1
}

handle_signal() {
  local signal_name="$1"
  local exit_code="$2"
  E2E_FINAL_STATUS="INTERRUPTED_${signal_name}"
  exit "${exit_code}"
}

trap 'write_e2e_summary "$?"' EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

{
  echo "[E2E] Run note: ${RUN_NOTE}"
  echo "[E2E] Run dir: ${RUN_DIR}"
  echo "[E2E] Status: RUNNING"
  echo "[E2E] Restart count: ${RESTART_COUNT}"
  echo "[E2E] Start time: ${E2E_START_TS}"
  echo "[E2E] Summary file initialized; final wall-clock will be written on exit."
} > "${E2E_SUMMARY_PATH}"

while true; do
  GPU_ASSIGNMENT=""
  NUM_GPUS=0
  RESTART_SOURCE_PATH=""

  # 🔧 Generate unique NCCL port for each restart iteration to avoid socket reuse conflicts
  # Use larger port spacing (100) to ensure TCP TIME_WAIT (60s) doesn't block port reuse
  # RESTART_COUNT=0: 31200 (first run)
  # RESTART_COUNT=1: 31300 (failover restart #1, avoids TIME_WAIT overlap)
  # RESTART_COUNT=2: 31400 (failover restart #2), etc.
  if [[ -n "${E2E_NCCL_BASE_PORT:-}" ]]; then
    _port_base="${E2E_NCCL_BASE_PORT}"
    echo "[E2E] Using preset NCCL base port: ${_port_base}"
  else
    _port_base=$((31200 + (RESTART_COUNT * 100)))
    echo "[E2E] 🔌 Port allocation: RESTART_COUNT=${RESTART_COUNT}, computed port=${_port_base} (spacing=100 to avoid TIME_WAIT)"
    if [[ ${RESTART_COUNT} -gt 0 ]]; then
      echo "[E2E] ✅ Failover restart iteration ${RESTART_COUNT}: Using NCCL port offset for clean socket state"
    fi
  fi

  if [[ -f "${SOFT_RESTART_CONFIG_PATH}" || -f "${HARD_RESTART_CONFIG_PATH}" || -f "${LEGACY_RESTART_CONFIG_PATH}" ]]; then
    mapfile -t _gpu_meta < <(DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS}" python - "${SOFT_RESTART_CONFIG_PATH}" "${HARD_RESTART_CONFIG_PATH}" "${LEGACY_RESTART_CONFIG_PATH}" <<'PY'
import json
import os
import sys

def parse_default_visible_gpus():
    raw = os.environ.get("DEFAULT_VISIBLE_GPUS", "")
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except Exception:
            continue
    return vals

def read_gpu_assignment(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        gpu_assignment = cfg.get('partition', {}).get('gpu_assignment', [])
        if isinstance(gpu_assignment, list) and gpu_assignment:
            return [int(x) for x in gpu_assignment]
    except Exception:
        return None
    return None

candidates = []
default_visible = parse_default_visible_gpus()
for p in sys.argv[1:]:
    if os.path.isfile(p):
        gpus = read_gpu_assignment(p)
        if gpus:
            # REPLAN payload often stores local indices (0..N-1) relative to current visibility.
            # Map those local indices back to physical GPU ids from DEFAULT_VISIBLE_GPUS.
            if default_visible and all(0 <= g < len(default_visible) for g in gpus):
                gpus = [default_visible[g] for g in gpus]
            candidates.append((os.path.getmtime(p), gpus, p))

if candidates:
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, gpus, src = candidates[0]
    print(','.join(str(x) for x in gpus))
    print(len(gpus))
    print(src)
else:
    print('')
    print(0)
    print('')
PY
)
    GPU_ASSIGNMENT="${_gpu_meta[0]:-}"
    NUM_GPUS="${_gpu_meta[1]:-0}"
    RESTART_SOURCE_PATH="${_gpu_meta[2]:-}"
  fi

  if [[ -n "${GPU_ASSIGNMENT}" && "${NUM_GPUS}" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ASSIGNMENT}"
    echo "[E2E] Restart config detected (${RESTART_SOURCE_PATH}) -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, partitions=${NUM_GPUS}"
  else
    export CUDA_VISIBLE_DEVICES="${DEFAULT_VISIBLE_GPUS}"
    NUM_GPUS=$(python - <<'PY'
import os
v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
items = [x.strip() for x in v.split(',') if x.strip()]
print(len(items) if items else 1)
PY
)
    echo "[E2E] Fresh start/default -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, partitions=${NUM_GPUS}"
  fi

  # NCCL 환경 설정 (로컬 단일 노드 디버깅/교착 회피용)
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
  export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-lo}"
  echo "[E2E] Dist env: MASTER_IP=${E2E_MASTER_IP}, NCCL_DEBUG=${NCCL_DEBUG}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE}, NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}, TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME}"

  # IMPORTANT:
  # TSPipe 자체가 단일 primary 프로세스에서 내부 worker/NCCL/RPC를 초기화합니다.
  # torchrun으로 외부 다중 프로세스를 추가하면 rank 충돌과 wait_ready 교착이 발생할 수 있어
  # E2E 런처는 항상 단일 python 프로세스로 train_kd.py를 실행합니다.

  # Note: NCCL port is now allocated per-restart inside the loop (see below)
  
  # Export failover scenario injection if specified (for SlowdownDetector intra-batch slowdown injection)
  # Note: Must re-export here because subprocess env vars don't propagate to script context
  export FAILOVER_INJECT_SCENARIO="${FAILOVER_INJECT_SCENARIO:=}"
  export FAILOVER_INJECT_GPU="${FAILOVER_INJECT_GPU:=}"
  export FAILOVER_INJECT_RATIO="${FAILOVER_INJECT_RATIO:=}"
  export FAILOVER_TEST_FAST_GATES="${FAILOVER_TEST_FAST_GATES:=}"
  
  # 기본 slowdown 지속 시간 게이트를 10초로 고정
  # (FAILOVER_SLOWDOWN_THRESHOLD_SEC를 외부에서 지정하면 그 값을 우선 사용)
  if [[ -z "${FAILOVER_SLOWDOWN_THRESHOLD_SEC:-}" ]]; then
    export FAILOVER_SLOWDOWN_THRESHOLD_SEC="10.0"
    echo "[E2E] Using default slowdown threshold: 10.0s"
  fi
  
  if [[ -n "${FAILOVER_INJECT_SCENARIO}" ]]; then
    echo "[E2E] Failover scenario injection enabled: ${FAILOVER_INJECT_SCENARIO}"
  fi
  
  set +e
  echo "[DEBUG] NCCL PORT: ${_port_base}, RESTART_COUNT: ${RESTART_COUNT}"  # Debug log
  PYTORCH_DISTRIBUTED_NCCL_START_PORT=${_port_base} \
  FAILOVER_INJECT_SCENARIO="${FAILOVER_INJECT_SCENARIO}" \
  FAILOVER_INJECT_GPU="${FAILOVER_INJECT_GPU}" \
  FAILOVER_INJECT_RATIO="${FAILOVER_INJECT_RATIO}" \
  FAILOVER_SLOWDOWN_THRESHOLD_SEC="${FAILOVER_SLOWDOWN_THRESHOLD_SEC:-}" \
  FAILOVER_TEST_FAST_GATES="${FAILOVER_TEST_FAST_GATES:-}" \
  python benchmarks/soft_target/train_kd.py \
    --tspipe-enable \
    --tspipe-config=benchmarks/soft_target/tspipe.yaml \
    --ip="${E2E_MASTER_IP}" \
    --rank=0 \
    --num-nodes=1 \
    --save_root "${BASE_SAVE_ROOT}" \
    --note "${RUN_NOTE}" \
    "$@"
  EXIT_CODE=$?
  set -e

  echo "[E2E] train_kd.py exited with code ${EXIT_CODE}"

  if [[ "${EXIT_CODE}" -eq 0 ]]; then
    E2E_FINAL_STATUS="COMPLETED"
    echo "[E2E] Training completed normally. Exiting launcher."
    break
  fi

  if [[ "${EXIT_CODE}" -eq "${FAILOVER_EXIT_CODE}" ]]; then
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[E2E] Failover restart requested (code ${FAILOVER_EXIT_CODE}), restart_count=${RESTART_COUNT}"

    if [[ "${MAX_RESTARTS}" -gt 0 && "${RESTART_COUNT}" -ge "${MAX_RESTARTS}" ]]; then
      echo "[E2E] Reached MAX_RESTARTS=${MAX_RESTARTS}. Stopping launcher."
      E2E_FINAL_STATUS="MAX_RESTARTS_REACHED"
      exit 1
    fi

    sleep 1
    continue
  fi

  echo "[E2E] Unexpected failure exit code ${EXIT_CODE}. Stopping launcher."
  E2E_FINAL_STATUS="FAILED"
  exit "${EXIT_CODE}"
done
