#!/bin/bash
# Quick Failover 실험 실행 스크립트
# 교수님께 바로 보고할 수 있는 결과 생성

echo "🚀 TSPipe Failover 실험 시작 (자동 GPU 할당)"
echo "================================================"

# 실험 환경 정보
echo "📋 실험 환경:"
echo "  - 호스트: $(hostname)"
echo "  - 날짜: $(date)"
echo "  - GPU 개수: $(nvidia-smi -L | wc -l)"
echo ""

# 현재 GPU 상태 확인 및 자동 할당
echo "🔍 현재 GPU 상태 확인 및 자동 할당:"
python3 gpu_auto_allocator.py

echo ""
echo "================================================"

# Setup environment
echo "🔧 환경 설정 중..."
source /home/wisekhy/miniconda3/bin/activate && conda activate tspipe
export PYTHONPATH="/home/wisekhy/tspipe/Synapse-private:$PYTHONPATH"
cd /home/wisekhy/tspipe/Synapse-private

# 이 실험에서는 GPU 0,3,4,6만 사용하도록 고정
export CUDA_VISIBLE_DEVICES="0,3,4,6"
echo "🎯 CUDA_VISIBLE_DEVICES 고정: ${CUDA_VISIBLE_DEVICES}"

# NCCL 환경 설정 (로컬 단일 노드 디버깅/교착 회피용)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
echo "🔧 NCCL env: NCCL_DEBUG=${NCCL_DEBUG}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE}, NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

# 기본 실험 실행 (자동 GPU 할당 사용)
echo "🔬 기본 Failover 실험 실행 중 (GPU 0,3,4,6 고정, 자동 GPU 할당 정보만 출력)..."
python run_failover_experiment.py \
    --experiment-type basic \
    --auto-gpu \
    --output-dir ./failover_results \
    --verbose

echo ""
echo "================================================"
echo "✅ 기본 실험 완료"

# 실험 결과 디렉토리 확인
LATEST_RESULT=$(find ./failover_results -name "failover_basic_*" -type d | sort | tail -1)

if [ -n "$LATEST_RESULT" ]; then
    echo "📊 실험 결과 위치: $LATEST_RESULT"
    echo ""
    echo "📁 생성된 파일들:"
    ls -la "$LATEST_RESULT"
    
    echo ""
    echo "📈 그래프 파일들:"
    if [ -d "$LATEST_RESULT/plots" ]; then
        ls -la "$LATEST_RESULT/plots"
    else
        echo "  그래프 파일이 생성되지 않았습니다."
    fi
    
    echo ""
    echo "🎯 교수님께 보고할 주요 파일들:"
    echo "  1. 실험 요약: $LATEST_RESULT/experiment_summary.json"
    echo "  2. 실험 로그: $LATEST_RESULT/experiment.log"
    echo "  3. 성능 데이터: $LATEST_RESULT/performance.jsonl"
    echo "  4. Failover 이벤트: $LATEST_RESULT/failover_events.jsonl"
    echo "  5. 성능 그래프: $LATEST_RESULT/plots/"
    
    # 간단한 결과 요약 출력
    echo ""
    echo "📋 실험 결과 요약:"
    if [ -f "$LATEST_RESULT/experiment_summary.json" ]; then
        echo "  - 실험 요약 파일 존재: ✅"
        # JSON에서 주요 정보 추출 (jq가 있다면)
        if command -v jq &> /dev/null; then
            echo "  - 실험 시간: $(jq -r '.experiment_info.duration_seconds' "$LATEST_RESULT/experiment_summary.json")초"
            echo "  - 총 실패 횟수: $(jq -r '.failover_statistics.total_failures' "$LATEST_RESULT/experiment_summary.json")"
            echo "  - 평균 복구 시간: $(jq -r '.failover_statistics.avg_recovery_time_ms' "$LATEST_RESULT/experiment_summary.json")ms"
        fi
    else
        echo "  - 실험 요약 파일 생성 실패: ❌"
    fi
    
else
    echo "❌ 실험 결과를 찾을 수 없습니다."
fi

echo ""
echo "================================================"

# 고급 실험도 실행할지 물어보기
echo ""
read -p "🔬 고급 실험 (다중 GPU 실패)도 실행하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔬 고급 Failover 실험 실행 중..."
    python run_failover_experiment.py \
        --experiment-type advanced \
        --auto-gpu \
        --output-dir ./failover_results \
        --verbose
    
    echo "✅ 고급 실험 완료"
    
    # 최신 결과 확인
    LATEST_ADVANCED_RESULT=$(find ./failover_results -name "failover_advanced_*" -type d | sort | tail -1)
    if [ -n "$LATEST_ADVANCED_RESULT" ]; then
        echo "📊 고급 실험 결과: $LATEST_ADVANCED_RESULT"
    fi
fi

echo ""
echo "================================================"
echo "🎉 모든 실험 완료!"
echo ""
echo "📋 결과 요약:"
echo "  - 기본 실험 결과: $LATEST_RESULT"
if [ -n "$LATEST_ADVANCED_RESULT" ]; then
    echo "  - 고급 실험 결과: $LATEST_ADVANCED_RESULT"
fi
echo ""
echo "📧 교수님께 보고 시 첨부할 파일들:"
echo "  1. 이 스크립트의 전체 출력"
echo "  2. 각 실험 디렉토리의 모든 내용"
echo "  3. 특히 experiment_summary.json과 plots/ 폴더"
echo ""
echo "✅ 실험 스크립트 실행 완료"