# TSPipe GPU Failover 실험 시스템

이 시스템은 TSPipe에서 GPU 실패 상황에 대한 자동 failover 및 복구 기능을 테스트합니다.

## 🎯 실험 목적

1. **GPU 실패 감지**: GPU 하드웨어 실패 자동 감지
2. **자동 복구**: 실패한 GPU를 제외하고 시스템 재구성
3. **동적 재분할**: 남은 GPU로 모델 파티션 최적화
4. **성능 모니터링**: 실패 및 복구 과정의 오버헤드 측정

## 📁 파일 구조

```
tspipe/
├── gpu_health_monitor.py      # GPU 상태 모니터링 시스템
├── failover_logger.py         # 실험 로깅 및 결과 분석
├── tspipe.py                  # 메인 TSPipe 클래스 (failover 기능 추가)
├── gpu_worker.py              # GPU Worker (헬스체크 기능 추가)
├── profiler.py                # 프로파일러 (failover 대응 기능)
├── scheduler.py               # 스케줄러 (동적 재분할 기능)
├── run_failover_experiment.py # 상세 실험 실행 스크립트
├── run_quick_failover_test.sh # 빠른 실험 실행 스크립트
└── README_FAILOVER.md         # 이 파일
```

## 🚀 빠른 실행 방법

### 방법 1: 원클릭 실행 (추천)

```bash
cd /home/wisekhy/tspipe/Synapse-private
./run_quick_failover_test.sh
```

### 방법 2: Python 스크립트 직접 실행

```bash
cd /home/wisekhy/tspipe/Synapse-private

# 기본 실험 (GPU 1개 실패)
python run_failover_experiment.py --experiment-type basic --target-gpu 4

# 고급 실험 (GPU 2개 연쇄 실패)
python run_failover_experiment.py --experiment-type advanced --target-gpu 4 --second-failure-gpu 5

# 프로파일링 오버헤드 측정
python run_failover_experiment.py --experiment-type profiling_overhead
```

### 방법 3: 기존 TSPipe에 failover 기능 추가하여 실행

```bash
cd /home/wisekhy/tspipe/Synapse-private/benchmarks/soft_target

python train_kd.py \
    --img_root=/nas-ssd/datasets/imagenet2012/imagenet \
    --save_root=./results/st/ \
    --t_model=./results/base/base-i100-vit-large/model_best.pth.tar \
    --s_init=./results/base/base-i100-resnet152/initial_r152.pth.tar \
    --kd_mode=st --lambda_kd=0.1 --t_name=vit_large --s_name=resnet152 \
    --T=4.0 --data_name=imagenet100 --num_class=100 --batch_size=16 \
    --tspipe-enable --tspipe-config=tspipe.yaml --num-node=1 --rank=0 --ip=localhost \
    --note=failover-test \
    --failover-enable \
    --failover-experiment="manual_test" \
    --backup-gpus=5,6 \
    --target-fail-gpu=4 \
    --fail-after-batches=10
```

## ⚙️ 실험 설정

### 현재 서버 설정
- **총 GPU**: 7개 (0~6번)
- **실험 사용 GPU**: 0, 1, 2, 3번 (4개)
- **백업 GPU**: 5, 6번 (다른 사용자가 안 쓰는 경우)
- **실패 시뮬레이션 대상**: 4번 GPU (다른 사용자가 안 쓰는 GPU)

### 실험 매개변수
- **헬스체크 주기**: 3-5초
- **실패 감지 방법**: CUDA 컨텍스트 테스트, 메모리 할당 테스트, nvidia-smi
- **복구 전략**: 백업 GPU 사용 → 남은 GPU로 재분할
- **프로파일링**: 1-2초 간격으로 성능 메트릭 수집

## 📊 실험 결과

실험 실행 후 `./failover_results/` 디렉토리에 다음 파일들이 생성됩니다:

### 1. 실험 요약 파일
- `experiment_summary.json`: 전체 실험 요약
- `experiment.log`: 상세 실험 로그

### 2. 성능 데이터
- `performance.jsonl`: 실시간 성능 메트릭 (CPU, GPU 사용률, 메모리)
- `failover_events.jsonl`: Failover 이벤트 상세 기록

### 3. 시각화 결과
- `plots/cpu_performance.png`: CPU 사용률 그래프
- `plots/gpu_memory.png`: GPU 메모리 사용률 그래프

## 🔬 실험 시나리오

### 기본 실험 (basic)
1. TSPipe 시작 (4개 GPU 사용)
2. 정상 동작 확인 (5 배치)
3. GPU 4번 실패 시뮬레이션
4. 자동 failover 및 복구 과정
5. 복구 후 정상 동작 확인 (3 배치)

### 고급 실험 (advanced)
1. TSPipe 시작
2. 정상 동작 확인
3. 첫 번째 GPU 실패 (4번)
4. 복구 과정
5. 두 번째 GPU 실패 (5번) - 연쇄 실패
6. 최종 복구 과정

### 오버헤드 실험 (profiling_overhead)
1. 프로파일링 비활성화 상태로 성능 측정
2. 프로파일링 활성화 상태로 성능 측정
3. 오버헤드 계산 및 분석

## 📈 성능 지표

### 측정 메트릭
- **실패 감지 시간**: GPU 실패부터 감지까지의 시간
- **복구 시간**: 실패 감지부터 시스템 재구성 완료까지의 시간
- **처리량 변화**: 실패 전후의 배치 처리 속도 변화
- **리소스 사용률**: CPU, GPU, 메모리 사용률 변화
- **프로파일링 오버헤드**: 모니터링 시스템의 성능 영향

### 예상 결과
- **실패 감지 시간**: 3-10초 (헬스체크 주기)
- **복구 시간**: 10-30초 (모델 재분할 + 프로세스 재시작)
- **처리량 감소**: 일시적으로 20-40% 감소 후 복구
- **오버헤드**: 1-5% (정상 상태)

## 🚨 주의사항

### 안전 고려사항
1. **다른 사용자 방해 금지**: GPU 4,5,6번만 실험에 사용
2. **실험 시간 제한**: 각 실험은 5-10분 내로 완료
3. **메모리 관리**: 실패 시뮬레이션 후 메모리 정리 자동 수행
4. **프로세스 정리**: 실험 종료 시 모든 프로세스 자동 정리

### 문제 해결
- **"GPU not available" 오류**: 다른 사용자가 사용 중인 GPU 피하기
- **메모리 부족 오류**: 실험 종료 후 `nvidia-smi` 확인 및 필요시 재부팅
- **프로세스 좀비**: `ps aux | grep python` 확인 후 수동 종료

## 🎯 교수님께 보고할 내용

### 1. 실험 실행 스크린샷
- 실험 실행 화면 캡처
- GPU 상태 변화 과정
- 복구 과정 로그

### 2. 실험 결과 파일
- `experiment_summary.json`: 핵심 성과 지표
- `plots/` 폴더: 성능 그래프들
- 실험 로그 중 중요한 부분

### 3. 성과 요약
- Failover 시스템 동작 확인 ✅
- 자동 GPU 실패 감지 ✅  
- 동적 모델 재분할 ✅
- 성능 오버헤드 측정 ✅
- 교육적 가치: 실제 산업 환경 문제 해결

### 4. 기술적 기여
- 기존 TSPipe에 failover 기능 추가
- 실시간 GPU 헬스 모니터링
- 동적 파티션 재구성 알고리즘
- 포괄적인 실험 로깅 시스템

## ❓ 문의사항

실험 실행 중 문제가 발생하면:

1. **즉시 확인**: 터미널 출력 및 오류 메시지
2. **로그 확인**: `failover_results/` 디렉토리의 로그 파일
3. **시스템 상태**: `nvidia-smi`, `ps aux` 명령어로 확인
4. **수동 정리**: 필요시 좀비 프로세스 수동 종료

---

**실험 실행 날짜**: $(date)  
**실험자**: wisekhy  
**TSPipe 버전**: Dynamic Programming Based Model Partitioning + Failover Extension