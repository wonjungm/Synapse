# TSPipe GPU Failover 시스템 구현 상세 문서

## 📋 프로젝트 개요

TSPipe 분산 딥러닝 파이프라인에 GPU 장애 상황에서의 자동 복구 및 동적 재구성 기능을 구현했습니다. 이 문서는 구현된 모든 코드와 실험 진행 방식을 상세히 설명합니다.

**목표**: GPU 실패 시 자동으로 감지하고, 백업 GPU로 복구하거나 모델을 동적으로 재분할하여 서비스 중단 없이 학습을 이어갈 수 있는 시스템 구축

---

## 🏗️ 시스템 아키텍처

### 핵심 구성요소
1. **GPU Health Monitor**: 실시간 GPU 상태 모니터링 및 실패 감지
2. **Failover Logger**: 실험 과정 및 성능 지표 상세 기록
3. **TSPipe Core**: Failover 로직이 통합된 메인 파이프라인
4. **GPU Worker**: 헬스체크 기능이 추가된 워커 프로세스
5. **Dynamic Scheduler**: GPU 실패 시 모델 재분할 알고리즘
6. **Auto GPU Allocator**: 사용 가능 GPU 자동 감지 및 할당

---

## 📂 구현된 파일별 상세 설명

### 1. `tspipe/gpu_health_monitor.py` - GPU 헬스 모니터링 시스템

```python
class GPUHealthMonitor:
    """실시간 GPU 상태 모니터링 및 실패 감지"""
```

**주요 기능**:
- 지정된 GPU들에 대해 5초 간격으로 헬스체크 수행
- CUDA 컨텍스트 테스트, 메모리 할당 테스트, nvidia-smi 상태 확인
- 실패 감지 시 콜백 함수를 통해 TSPipe에 즉시 알림
- 실패 시뮬레이션 기능 (테스트 목적)

**핵심 메서드**:
- `start_monitoring()`: 백그라운드 모니터링 시작
- `simulate_gpu_failure()`: 특정 GPU 실패 시뮬레이션
- `_check_gpu_health()`: 개별 GPU 상태 검사
- `get_status()`: 모든 GPU 현재 상태 반환

**통합 방식**: TSPipe 초기화 시 자동으로 시작되며, GPU 실패 감지 시 `_handle_gpu_failure()` 콜백 호출

### 2. `tspipe/failover_logger.py` - 실험 로깅 및 분석 시스템

```python
class FailoverExperimentLogger:
    """Failover 실험의 모든 과정을 상세히 기록하고 분석"""
```

**주요 기능**:
- 실험 시작/종료, GPU 실패 이벤트, 성능 지표를 구조화된 JSON 형태로 로깅
- 실시간 성능 메트릭 수집 (GPU 사용률, 메모리, 처리량 등)
- 실험 종료 시 자동으로 결과 분석 및 시각화 그래프 생성
- 교수님 보고용 요약 리포트 자동 생성

**생성되는 파일들**:
- `experiment.log`: 상세 텍스트 로그
- `experiment_summary.json`: 실험 요약 정보
- `performance.jsonl`: 시간별 성능 데이터
- `failover_events.jsonl`: GPU 실패 이벤트 로그
- `plots/`: 성능 그래프들 (GPU 사용률, 메모리, 처리량 등)

**핵심 메서드**:
- `log_gpu_failure()`: GPU 실패 이벤트 기록
- `start_metrics_collection()`: 실시간 메트릭 수집 시작
- `generate_summary()`: 실험 결과 요약 생성
- `create_performance_plots()`: 성능 분석 그래프 생성

### 3. `tspipe/tspipe.py` - Failover 통합 메인 클래스

기존 TSPipe 클래스에 Failover 기능이 완전 통합되었습니다.

**추가된 주요 기능**:
- `--enable-failover`: Failover 기능 활성화 옵션
- `--backup-gpus`: 백업 GPU 지정
- `--health-check-interval`: 헬스체크 간격 설정

**핵심 추가 메서드**:
```python
def _init_failover_system(self):
    """Failover 시스템 초기화"""
    
def _handle_gpu_failure(self, failed_gpu_id, error_info):
    """GPU 실패 시 복구 로직 실행"""
    
def _attempt_gpu_replacement(self, failed_gpu_id):
    """백업 GPU로 교체 시도"""
    
def _attempt_dynamic_repartition(self):
    """동적 모델 재분할 시도"""
```

**Failover 전략**:
1. **1차**: 백업 GPU가 있으면 즉시 교체
2. **2차**: 백업 GPU가 없으면 남은 GPU들에 모델 재분할
3. **3차**: 복구 불가능하면 안전하게 실험 종료

### 4. `tspipe/gpu_worker.py` - 워커 프로세스 헬스체크

GPU 워커 프로세스에 자체 헬스체크 기능을 추가했습니다.

**추가된 기능**:
- 워커별 주기적 자가진단 (GPU 메모리, CUDA 상태 등)
- 이상 감지 시 메인 프로세스에 알림
- Graceful shutdown 지원

### 5. `tspipe/profiler.py` - 프로파일러 Failover 지원

프로파일링 도중 GPU 실패가 발생해도 데이터 손실 없이 복구할 수 있도록 개선했습니다.

**추가된 기능**:
- Failover 이벤트 발생 시 프로파일링 데이터 즉시 저장
- 복구 후 프로파일링 재시작 기능
- Failover 오버헤드 측정 기능

### 6. `tspipe/scheduler.py` - 동적 재분할 스케줄러

GPU 실패 시 남은 GPU들에 모델을 효율적으로 재분배하는 알고리즘을 구현했습니다.

**핵심 기능**:
```python
def dynamic_repartition(self, available_gpus, original_split):
    """남은 GPU에 맞게 모델 재분할"""
```

**재분할 전략**:
- 레이어별 계산 복잡도 고려
- 메모리 사용량 균형 맞춤  
- 통신 오버헤드 최소화

### 7. `gpu_auto_allocator.py` - 자동 GPU 할당기

**4개 GPU 필수 조건** 구현으로 안전한 실험을 보장합니다.

**주요 기능**:
- nvidia-smi를 통한 실시간 GPU 사용률 확인
- 사용률 90% 미만의 GPU만 "사용가능"으로 판단
- 4개 GPU 미만이면 실험 자동 중단
- 사용가능한 GPU에 따른 실험 시나리오 자동 조정

**할당 시나리오**:
- **5개+ GPU**: 완전한 실험 (4개 실험용 + 1개+ 백업)
- **정확히 4개**: 최소 실험 (3개 실험용 + 1개 실패대상)
- **4개 미만**: 실험 불가능 (안전 중단)

### 8. `run_failover_experiment.py` - 메인 실험 스크립트

다양한 실험 시나리오를 자동으로 수행하는 통합 스크립트입니다.

**실험 타입**:
- `basic`: 단일 GPU 실패 + 백업으로 복구
- `advanced`: 연속 다중 GPU 실패
- `profiling_overhead`: Failover 오버헤드 측정

**주요 옵션**:
- `--auto-gpu`: 자동 GPU 할당 (권장)
- `--experiment-type`: 실험 시나리오 선택
- `--output-dir`: 결과 저장 경로

### 9. `run_quick_failover_test.sh` - 원클릭 실험 실행

팀원들이 쉽게 실험할 수 있도록 모든 과정을 자동화한 스크립트입니다.

**자동 수행 과정**:
1. Conda 환경 활성화 (`conda activate tspipe`)
2. GPIO 상태 확인 및 할당
3. 기본 실험 자동 실행  
4. 결과 파일 위치 안내
5. 선택적으로 고급 실험 추가 수행

---

## 🧪 실험 진행 방식

### 실험 전 준비사항
1. **환경 확인**: 4개 이상의 GPU가 사용가능한 상태인지 확인
2. **Conda 환경**: `conda activate tspipe` 필수
3. **권한 설정**: 실행 스크립트에 실행 권한 부여

### 실험 실행 방법

#### 방법 1: 원클릭 실행 (권장)
```bash
cd /home/wisekhy/tspipe/Synapse-private
./run_quick_failover_test.sh
```

#### 방법 2: 수동 실행
```bash
conda activate tspipe
cd /home/wisekhy/tspipe/Synapse-private

# GPU 상태 확인
python3 gpu_auto_allocator.py

# 실험 실행 (자동 GPU 할당)
python run_failover_experiment.py --experiment-type basic --auto-gpu
```

### 실험 시나리오별 상세 설명

#### 1. Basic Failover Test (기본 실험)
- **소요시간**: 약 15-20분
- **진행과정**:
  1. 4개 GPU로 TSPipe 파이프라인 시작
  2. 10분간 정상 동작 확인
  3. 지정된 GPU에 실패 시뮬레이션
  4. 백업 GPU로 자동 복구 확인
  5. 복구 후 3분간 안정성 테스트

#### 2. Advanced Failover Test (고급 실험)  
- **소요시간**: 약 25-30분
- **진행과정**:
  1. 기본 실험과 동일하게 시작
  2. 첫 번째 GPU 실패 + 복구
  3. 잠시 후 두 번째 GPU 연속 실패
  4. 동적 재분할 알고리즘 동작 확인
  5. 성능 영향도 측정

#### 3. Profiling Overhead Test
- **소요시간**: 약 10분
- **목적**: Failover 시스템이 추가되면서 발생하는 성능 오버헤드 측정

### 실험 결과 분석

각 실험 완료 후 다음 결과물들이 자동 생성됩니다:

#### 📊 교수님 보고용 주요 파일들
1. **`experiment_summary.json`**: 실험 전체 요약
   - 실험 설정, 총 소요시간, GPU 실패 횟수
   - 복구 성공률, 성능 영향도 등

2. **`failover_events.jsonl`**: GPU 실패 이벤트 상세 로그
   - 실패 감지 시간, 복구 시간, 복구 방법
   - 각 단계별 소요시간 등

3. **`plots/`**: 시각화 그래프들
   - GPU 사용률 변화
   - 메모리 사용량 트렌드  
   - 처리량(throughput) 변화
   - Failover 이벤트 타임라인

#### 📈 성능 지표 (KPI)
- **실패 감지 시간**: GPU 실패부터 감지까지의 지연시간
- **복구 완료 시간**: 감지부터 정상 동작 재개까지의 시간
- **성능 영향도**: Failover 전후 처리량 비교
- **복구 성공률**: 시도한 복구 중 성공한 비율

---

## 🔧 현재 상태 및 다음 단계

### ✅ 완료된 기능
1. GPU 실시간 헬스 모니터링 시스템
2. 자동 복구 로직 (백업 GPU 교체 + 동적 재분할)
3. 상세한 실험 로깅 및 결과 분석
4. 4개 GPU 필수 조건 안전장치
5. 자동 GPU 할당 및 실험 스크립트

### ⚠️ 현재 제약사항
- **서버 상황**: 현재 GPU 0만 사용가능(1개), 나머지는 다른 사용자가 사용중
- **실험 조건**: 4개 GPU 필수이므로 현재는 실험 불가능
- **대기 필요**: 다른 사용자의 작업 완료까지 대기 후 실험 필요

### 🔄 대기 중인 실험
```bash
# 서버 상태 체크
nvidia-smi

# 4개 이상 GPU 사용가능해지면 즉시 실험 가능
./run_quick_failover_test.sh
```

### 📋 팀 작업 분담 제안
1. **실험 담당**: GPU 사용가능 시간대에 실험 수행 및 결과 수집
2. **분석 담당**: 생성된 로그 데이터 추가 분석 및 리포트 작성  
3. **개선 담당**: 실험 결과 바탕으로 알고리즘 성능 튜닝

---

## 📞 문의 및 트러블슈팅

### 자주 발생하는 문제
1. **ModuleNotFoundError**: `conda activate tspipe` 확인
2. **GPU 부족**: 4개 GPU 사용가능할 때까지 대기
3. **권한 오류**: `chmod +x run_quick_failover_test.sh` 실행

### 긴급 상황 대응
- 실험 중단: `Ctrl+C`
- 프로세스 강제 종료: `nvidia-smi` 확인 후 해당 프로세스 kill
- 로그 확인: `failover_results/` 디렉토리의 최신 실험 폴더

### 팀원 연락처
- 실험 관련 문의: [담당자 정보]
- 기술적 이슈: [개발자 정보]
- 결과 분석: [분석가 정보]

---

*이 문서는 TSPipe GPU Failover 시스템의 완전한 구현 상태를 기록한 것입니다. 추가 질문이나 개선사항이 있으면 언제든 연락주세요.*