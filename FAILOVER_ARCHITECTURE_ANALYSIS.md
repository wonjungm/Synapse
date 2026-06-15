# TSPipe Failover Architecture - Quick Analysis

**작성일**: 2026년 3월 9일  

## 요약

✅ **현재**: Failover 인프라 기본 완성 (감지/응답)  
❌ **부재**: 정책 계층 (KEEP/REPLAN/DEGRADE)  
🎯 **결론**: 정책 추가 구현 가능한 준비 완료

---

## 1️⃣ 학습 루프 구조

**진입점**: `train_kd_profiling.py` → `TSPipe.start_pipeline()` → `_thread_scheduler()` (별도 스레드)

**흐름**:
```
for each batch:
  ├─ task_scheduler.schedule_generator() → 스케줄 생성
  ├─ GPU에 task 디스패치 (forward, backward, optimizer)
  ├─ _record_step_metric() → step time 기록
  ├─ _save_healthy_checkpoint() (주기적, 기본 20배치)
  └─ GPU failure 시뮬레이션 체크
```

**주요 흐름**: Teacher Forward → Student Forward → Loss → Backward → Optimizer Step → Momentum Update

**핵심 변수**: `batch_count`, `forward_complete_batch_idx`, `_shutdown_scheduler

---

## 2️⃣ Partition 구조

**표현**: YAML `model_split`에 정의
```yaml
model_split:
  online:  [6, 6, 6, 5]    # Student 레이어 분할
  target:  [5, 5, 4, 4]    # Teacher 레이어 분할
```

**계산**: `TSPipe.split_module()` → Sequential 객체 리스트 생성

**특징**:
- ✅ 초기화 시 계산 (YAML 기반)
- ❌ 런타임 repartition 불가 (재시작 필요)
- ✅ K-1 GPU failover 시 새 partition 생성 (`_calculate_new_partition_config()`, 균등분할)

**Failover 시 처리**:
```
GPU 실패 감지
  ↓
_prepare_k_minus_1_restart() 호출
  ├─ 가용 GPU 목록 계산
  ├─ restart_config.json 생성 (새 partition 정보 포함)
  └─ 외부 스크립트가 재시작 시 사용
```

**저장 변수**: `original_partition_config`, `current_partition_config`

---

## 3️⃣ Failover 메커니즘

**감지**: 
- `GPUHealthMonitor` (주기적 GPU 체크, 기본 5초)
- `ProcessHealthMonitor` (worker 프로세스 상태)

**발생 시 동작**:
```
GPU/프로세스 실패 감지
  ↓
_on_gpu_failure() / _on_process_failure() 콜백
  ↓
_emergency_shutdown_and_failover() 호출
  ├─ 학습 루프 강제 중단 (_shutdown_scheduler = True)
  ├─ 실패 worker 종료
  ├─ 분산 통신 정리 (NCCL/RPC 종료)
  ├─ 비상 + 정상 체크포인트 저장
  ├─ restart_config.json 생성 (K-1 GPU용)
  └─ os._exit(1) 프로세스 종료
```

**외부 처리**: `run_failover_experiment.py` 스크립트가 restart_config.json 읽고 K-1 GPU로 재시작

**옵션**:
```bash
--failover-enable                    # Failover 활성화
--target-fail-gpu N                  # N번 GPU 강제 실패
--fail-after-batches M               # M 배치 후 실패
--healthy-checkpoint-interval K      # K 배치마다 체크포인트 (기본 20)
```

**부재 기능**: ❌ Slowdown 감지, ❌ 동적 정책, ❌ 부분 실패 대응

---

## 4️⃣ Checkpoint 메커니즘

| 타입 | 파일명 | 주기 | 내용 |
|------|--------|------|------|
| **정상** | `healthy_checkpoint_latest.pth` | 주기적 (기본 20배치) | `{model_state_dict, batch_count, timestamp}` |
| **비상** | `emergency_checkpoint.pth` | 장애 직후 | `state_dict만` |
| **설정** | `restart_config.json` | 장애 직후 | `{failed_gpu, available_gpus, partition_config, paths}` |

**복구 흐름**:
```
Failover → checkpoint + restart_config 생성 → os._exit(1)
  ↓
외부 스크립트 (run_experiment0b_failover.py)
  ├─ restart_config.json 읽음
  ├─ 새 YAML 파일 생성 (K-1 GPU용)
  └─ train_kd_profiling.py 재시작
      └─ --resume-checkpoint=healthy_checkpoint_latest.pth
```

**복구 내용**: ✅ 모델 파라미터, ✅ 배치 카운트, ❌ Optimizer 상태, ❌ Scheduler 상태

**메타데이터**: `exp0b_failover_save_events.jsonl`에 checkpoint 정보 기록

---

## 5️⃣ Runtime 모니터링

**수집 정보**:

1. **Step Timing** (매 배치)
   ```json
   {"event_type": "step_timing", "step_id": 1050, "step_time_ms": 45.2, "batch_count": 1050}
   ```

2. **Checkpoint Impact** (checkpoint 후)
   ```json
   {"event_type": "checkpoint_spike_observed", "pre_step_time_ms": 45.2, 
    "post_step_time_ms": 67.8, "delta_ms": 22.6, "ratio_post_over_pre": 1.50}
   ```

**저장 파일**:
- `exp0b_failover_step_metrics.jsonl` (매 배치의 step time)
- `exp0b_failover_save_events.jsonl` (checkpoint & slowdown 정보)

**Slowdown 분석**: ✅ `ratio_post_over_pre`로 직접 계산 가능

**부재**: ❌ Per-GPU timing, ❌ Stage별 timing (추가 구현 필요)

---

## 6️⃣ 정책 삽입 가능 지점

**1순위 (권장)**: Checkpoint 후 정책 평가
```python
def _save_healthy_checkpoint(self):
    torch.save({...}, checkpoint_path)
    # ⭐ 정책 평가 hook 추가
    self._evaluate_and_apply_runtime_policy()
```

**2순위**: 학습 루프 내 주기적 평가
```python
def _thread_scheduler(self):
    for schedules in self.task_scheduler.schedule_generator():
        if self.batch_count % POLICY_EVAL_INTERVAL == 0:
            # 정책 평가
```

**3순위**: 별도 모니터 스레드 (고급)

**장점**: Checkpoint 직후이므로 재시작 준비 완료, 정책 실행 비용 최소화

**필수 신규 메서드**:
```python
def _evaluate_runtime_policy(self) -> str:
    """현재 상태 분석 → 'KEEP'/'REPLAN'/'DEGRADE' 반환"""
    recent_timings = self._get_recent_step_timings(window=20)
    slowdown_ratio = mean(recent_timings) / baseline
    if slowdown_ratio > THRESHOLD_DEGRADE: return 'DEGRADE'
    elif slowdown_ratio > THRESHOLD_REPLAN: return 'REPLAN'
    else: return 'KEEP'

def _apply_runtime_policy(self, policy: str):
    """정책 실행"""
    if policy == 'KEEP': return
    elif policy == 'REPLAN': 
        new_config = self._compute_optimal_partition()
        self._trigger_controlled_restart(new_config)
```

**문제점**: 현재 구조는 failure → immediate os._exit(1), 정책 실행 제한적

---

## 7️⃣ 구현 로드맵

**Phase 1 (1-2주)**: 기본 정책 평가
- Runtime metrics 수집 및 분석
- Threshold 정의 (KEEP/REPLAN/DEGRADE)
- `_evaluate_runtime_policy()` 구현

**Phase 2 (2-3주)**: 정책 실행
- `_apply_runtime_policy()` 구현
- Checkpoint 후 정책 평가 hook 삽입
- 정책 효과 로깅

**Phase 3 (미래)**: 고급 기능
- 핫 리스타트 (무중단 repartition)
- 머신러닝 기반 정책 선택

---

## 📊 시스템 구조

```
train_kd_profiling.py
    ↓
TSPipe.__init__() → split_module, 스케줄러 생성, Failover 초기화
    ↓
_thread_scheduler() [메인 루프]
├─ task 디스패치 & GPU 실행
├─ _record_step_metric() [⭐ 데이터 수집]
├─ _save_healthy_checkpoint() [주기적]
│  └─ [🎯 정책 평가 위치]
└─ GPU failure 체크

[병렬] Failover 모니터 스레드
├─ GPUHealthMonitor
└─ ProcessHealthMonitor
   └─ failure 감지 → _emergency_shutdown_and_failover()
      ├─ checkpoint 저장
      ├─ restart_config.json 생성
      └─ os._exit(1)

외부 스크립트 개입 (run_failover_experiment.py)
└─ K-1 GPU로 재시작 (--resume-checkpoint)
```

---

## 📝 핵심 파일

| 기능 | 파일 | 메서드 |
|------|------|--------|
| 학습 루프 | `tspipe/tspipe.py` | `_thread_scheduler()` |
| 스케줄링 | `tspipe/scheduler.py` | `schedule_generator()` |
| Partition | `tspipe/tspipe.py` | `split_module()` |
| GPU 감지 | `tspipe/gpu_health_monitor.py` | `GPUHealthMonitor` |
| Checkpoint | `tspipe/tspipe.py` | `_save_healthy_checkpoint()` |
| Runtime 지표 | `tspipe/tspipe.py` | `_record_step_metric()` |
| Failover 처리 | `tspipe/tspipe.py` | `_emergency_shutdown_and_failover()` |
| 외부 재시작 | `run_failover_experiment.py` | `_attempt_k_minus_1_restart()` |

---

## 🎯 결론

### ✅ 현재 준비됨
- ✅ Failover 기본 인프라
- ✅ Checkpoint 메커니즘
- ✅ Runtime metrics
- ✅ K-1 재분할 기본 로직

### ❌ 부재
- ❌ Slowdown 감지
- ❌ 동적 정책 선택
- ❌ Graceful degradation

### 🚀 추천 다음 단계
1. `_evaluate_runtime_policy()` 구현
2. `_apply_runtime_policy()` 구현  
3. Checkpoint 후 정책 평가 hook 삽입

