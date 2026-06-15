#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_failover_with_phase0_gated_bgload.sh \
    --bg-profile <profile> \
    [--run-note <run_note>] \
    [--bg-gpu <gpu_id>] \
    [--bg-log <path>] \
    [--base-save-root <dir>] \
    [--release-pattern <text>] \
    [--release-timeout-sec <sec>] \
    [--bg-duration-sec <sec>] \
    [--bg-post-release-delay-sec <sec>] \
    [--bg-size <n>] [--bg-num-streams <n>] [--bg-loop-count <n>] \
    [--bg-inner-loops <n>] [--bg-cycle-sleep-sec <sec>] \
    -- <run_e2e_failover.sh args...>

Profiles:
  twostream_b128
  strongplus_b64
  strong_b256
  strongscaled_b512

Example:
  scripts/run_failover_with_phase0_gated_bgload.sh \
    --bg-profile strongscaled_b512 \
    --run-note "e2e_failover_bgload_gpu3_phase0_b512_$(date +%Y%m%d_%H%M%S)" \
    --bg-log /tmp/bgload_failover_gpu3_phase0_b512.log \
    -- \
    --img_root=/workspace/datasets/imagenet \
    --data_name=imagenet100 \
    --t_name=vit_large \
    --s_name=resnet152 \
    --kd_mode=st \
    --lambda_kd=0.1 \
    --t_model=/workspace/Synapse/Synapse/benchmarks/soft_target/results/base/base-i100-vit-large/model_best.pth.tar \
    --s_init=/workspace/Synapse/Synapse/benchmarks/soft_target/results/base/base-i100-resnet152/initial_r152.pth.tar \
    --batch_size=512 \
    --num_class=100 \
    --epochs=1 \
    --max-steps-per-epoch=0 \
    --tspipe-enable \
    --tspipe-config=benchmarks/soft_target/tspipe.yaml \
    --inject-slowdown-gpu=3

This wrapper:
  1. prewarms the bgload process and leaves it armed
  2. tails results/${RUN_NOTE}/log.txt from the current end-of-file
  3. releases bgload as soon as "Phase-0 baseline frozen" appears
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BG_PROFILE="${BG_PROFILE:-}"
BG_GPU="${BG_GPU:-3}"
BG_LOG="${BGLOG:-}"
BG_DURATION_SEC="${BG_DURATION_SEC:-4000}"
BG_POST_RELEASE_DELAY_SEC="${BG_POST_RELEASE_DELAY_SEC:-0}"
BG_SIZE="${BG_SIZE:-}"
BG_NUM_STREAMS="${BG_NUM_STREAMS:-}"
BG_LOOP_COUNT="${BG_LOOP_COUNT:-}"
BG_INNER_LOOPS="${BG_INNER_LOOPS:-}"
BG_CYCLE_SLEEP_SEC="${BG_CYCLE_SLEEP_SEC:-}"
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-./results}"
RUN_NOTE="${RUN_NOTE:-}"
RELEASE_PATTERN="${BGLOAD_RELEASE_PATTERN:-Phase-0 baseline frozen at step=}"
RELEASE_TIMEOUT_SEC="${BGLOAD_RELEASE_TIMEOUT_SEC:-3600}"
READY_TIMEOUT_SEC="${BGLOAD_READY_TIMEOUT_SEC:-180}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --bg-profile)
      BG_PROFILE="$2"
      shift 2
      ;;
    --bg-profile=*)
      BG_PROFILE="${1#*=}"
      shift
      ;;
    --run-note)
      RUN_NOTE="$2"
      shift 2
      ;;
    --run-note=*)
      RUN_NOTE="${1#*=}"
      shift
      ;;
    --base-save-root)
      BASE_SAVE_ROOT="$2"
      shift 2
      ;;
    --base-save-root=*)
      BASE_SAVE_ROOT="${1#*=}"
      shift
      ;;
    --bg-gpu)
      BG_GPU="$2"
      shift 2
      ;;
    --bg-gpu=*)
      BG_GPU="${1#*=}"
      shift
      ;;
    --bg-log)
      BG_LOG="$2"
      shift 2
      ;;
    --bg-log=*)
      BG_LOG="${1#*=}"
      shift
      ;;
    --release-pattern)
      RELEASE_PATTERN="$2"
      shift 2
      ;;
    --release-pattern=*)
      RELEASE_PATTERN="${1#*=}"
      shift
      ;;
    --release-timeout-sec)
      RELEASE_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --release-timeout-sec=*)
      RELEASE_TIMEOUT_SEC="${1#*=}"
      shift
      ;;
    --bg-duration-sec)
      BG_DURATION_SEC="$2"
      shift 2
      ;;
    --bg-duration-sec=*)
      BG_DURATION_SEC="${1#*=}"
      shift
      ;;
    --bg-post-release-delay-sec)
      BG_POST_RELEASE_DELAY_SEC="$2"
      shift 2
      ;;
    --bg-post-release-delay-sec=*)
      BG_POST_RELEASE_DELAY_SEC="${1#*=}"
      shift
      ;;
    --bg-size)
      BG_SIZE="$2"
      shift 2
      ;;
    --bg-size=*)
      BG_SIZE="${1#*=}"
      shift
      ;;
    --bg-num-streams)
      BG_NUM_STREAMS="$2"
      shift 2
      ;;
    --bg-num-streams=*)
      BG_NUM_STREAMS="${1#*=}"
      shift
      ;;
    --bg-loop-count)
      BG_LOOP_COUNT="$2"
      shift 2
      ;;
    --bg-loop-count=*)
      BG_LOOP_COUNT="${1#*=}"
      shift
      ;;
    --bg-inner-loops)
      BG_INNER_LOOPS="$2"
      shift 2
      ;;
    --bg-inner-loops=*)
      BG_INNER_LOOPS="${1#*=}"
      shift
      ;;
    --bg-cycle-sleep-sec)
      BG_CYCLE_SLEEP_SEC="$2"
      shift 2
      ;;
    --bg-cycle-sleep-sec=*)
      BG_CYCLE_SLEEP_SEC="${1#*=}"
      shift
      ;;
    --)
      shift
      PASS_ARGS+=("$@")
      break
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${BG_PROFILE}" ]]; then
  echo "error: --bg-profile is required" >&2
  usage
  exit 2
fi

if [[ -z "${RUN_NOTE}" ]]; then
  RUN_NOTE="e2e_failover_bgload_${BG_PROFILE}_phase0_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "${BG_LOG}" ]]; then
  BG_LOG="/tmp/${RUN_NOTE}_bgload.log"
fi

RUN_DIR="${BASE_SAVE_ROOT}/${RUN_NOTE}"
TRAIN_LOG="${RUN_DIR}/log.txt"

mkdir -p "${RUN_DIR}"
GATE_DIR="$(mktemp -d "/tmp/${RUN_NOTE}_gate.XXXXXX")"
RELEASE_FILE="${GATE_DIR}/release"
READY_FILE="${GATE_DIR}/ready"
STARTED_FILE="${GATE_DIR}/started"
WATCH_LOG="${GATE_DIR}/watcher.log"

WATCH_PID=""
BG_PID=""

cleanup() {
  if [[ -n "${WATCH_PID}" ]]; then
    kill "${WATCH_PID}" 2>/dev/null || true
  fi
  if [[ -n "${BG_PID}" ]]; then
    kill "${BG_PID}" 2>/dev/null || true
  fi
  rm -rf "${GATE_DIR}"
}
trap cleanup EXIT INT TERM

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate tspipe
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "error: neither ${PYTHON_BIN} nor python3 is available" >&2
    exit 1
  fi
fi

EXISTING_LOG_SIZE=0
if [[ -f "${TRAIN_LOG}" ]]; then
  EXISTING_LOG_SIZE="$(stat -c%s "${TRAIN_LOG}" 2>/dev/null || echo 0)"
fi

BG_CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/gated_bgload.py"
  --profile "${BG_PROFILE}"
  --release-file "${RELEASE_FILE}"
  --ready-file "${READY_FILE}"
  --started-file "${STARTED_FILE}"
  --duration-sec "${BG_DURATION_SEC}"
  --post-release-delay-sec "${BG_POST_RELEASE_DELAY_SEC}"
)

if [[ -n "${BG_SIZE}" ]]; then
  BG_CMD+=(--size "${BG_SIZE}")
fi
if [[ -n "${BG_NUM_STREAMS}" ]]; then
  BG_CMD+=(--num-streams "${BG_NUM_STREAMS}")
fi
if [[ -n "${BG_LOOP_COUNT}" ]]; then
  BG_CMD+=(--loop-count "${BG_LOOP_COUNT}")
fi
if [[ -n "${BG_INNER_LOOPS}" ]]; then
  BG_CMD+=(--inner-loops "${BG_INNER_LOOPS}")
fi
if [[ -n "${BG_CYCLE_SLEEP_SEC}" ]]; then
  BG_CMD+=(--cycle-sleep-sec "${BG_CYCLE_SLEEP_SEC}")
fi

echo "[gate] run note=${RUN_NOTE}"
echo "[gate] run dir=${RUN_DIR}"
echo "[gate] train log=${TRAIN_LOG}"
echo "[gate] bg profile=${BG_PROFILE} bg gpu=${BG_GPU} bg log=${BG_LOG}"
echo "[gate] release pattern=${RELEASE_PATTERN}"

CUDA_VISIBLE_DEVICES="${BG_GPU}" "${BG_CMD[@]}" >"${BG_LOG}" 2>&1 &
BG_PID=$!
echo "[gate] armed bgload pid=${BG_PID}"

READY_DEADLINE=$((SECONDS + READY_TIMEOUT_SEC))
while [[ ! -f "${READY_FILE}" ]]; do
  if ! kill -0 "${BG_PID}" 2>/dev/null; then
    echo "[gate] bgload exited before becoming ready; see ${BG_LOG}" >&2
    exit 1
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    echo "[gate] timed out waiting for bgload ready marker: ${READY_FILE}" >&2
    exit 1
  fi
  sleep 1
done
echo "[gate] bgload warmup complete; waiting for phase-0 freeze"

"${PYTHON_BIN}" - "${TRAIN_LOG}" "${RELEASE_FILE}" "${RELEASE_PATTERN}" "${EXISTING_LOG_SIZE}" "${RELEASE_TIMEOUT_SEC}" <<'PY' >"${WATCH_LOG}" 2>&1 &
import sys
import time
from pathlib import Path

log_path = Path(sys.argv[1])
release_path = Path(sys.argv[2])
pattern = sys.argv[3]
start_offset = int(sys.argv[4])
timeout_sec = float(sys.argv[5])

deadline = time.time() + timeout_sec
print(
    f"[watcher] watching {log_path} for pattern={pattern!r} starting at offset={start_offset}",
    flush=True,
)

while time.time() < deadline:
    if log_path.exists():
        break
    time.sleep(0.2)
else:
    print(f"[watcher] timeout waiting for log file: {log_path}", flush=True)
    sys.exit(2)

with log_path.open("r", encoding="utf-8", errors="replace") as f:
    current_size = log_path.stat().st_size
    if start_offset > current_size:
        start_offset = current_size
    f.seek(start_offset)

    while time.time() < deadline:
        line = f.readline()
        if not line:
            time.sleep(0.2)
            continue
        if pattern in line:
            release_path.write_text(
                f"released_at={time.time():.6f}\npattern={pattern}\nline={line}",
                encoding="utf-8",
            )
            print(f"[watcher] matched line: {line.strip()}", flush=True)
            print(f"[watcher] released bgload via {release_path}", flush=True)
            sys.exit(0)

print(f"[watcher] timeout waiting for release pattern in {log_path}", flush=True)
sys.exit(3)
PY
WATCH_PID=$!

export BASE_SAVE_ROOT
export RUN_NOTE

set +e
(
  cd "${REPO_ROOT}"
  bash ./run_e2e_failover.sh "${PASS_ARGS[@]}"
)
RUN_RC=$?
set -e

WATCH_RC=0
if [[ -n "${WATCH_PID}" ]]; then
  if kill -0 "${WATCH_PID}" 2>/dev/null; then
    if [[ ${RUN_RC} -ne 0 || ! -f "${RELEASE_FILE}" ]]; then
      kill "${WATCH_PID}" 2>/dev/null || true
    fi
  fi

  set +e
  wait "${WATCH_PID}"
  WATCH_RC=$?
  set -e

  if [[ ! -f "${RELEASE_FILE}" && ${WATCH_RC} -ne 0 ]]; then
    WATCH_RC=4
  fi
fi

echo "[gate] run_e2e_failover.sh exit code=${RUN_RC}"
echo "[gate] watcher exit code=${WATCH_RC}"
echo "[gate] watcher log=${WATCH_LOG}"
echo "[gate] bg log=${BG_LOG}"

if [[ -f "${STARTED_FILE}" ]]; then
  echo "[gate] bgload release confirmed"
else
  echo "[gate] warning: bgload start marker was not written" >&2
fi

if [[ ${RUN_RC} -ne 0 ]]; then
  exit "${RUN_RC}"
fi

if [[ ${WATCH_RC} -ne 0 ]]; then
  exit "${WATCH_RC}"
fi

exit 0
