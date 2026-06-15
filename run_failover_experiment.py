#!/usr/bin/env python3
"""
Failover 실험 실행 스크립트
GPU 실패 상황에서 TSPipe의 자동 복구 및 동적 재구성 실험

사용법:
    python run_failover_experiment.py --experiment-type basic
    python run_failover_experiment.py --experiment-type advanced --target-gpu 4
    python run_failover_experiment.py --auto-gpu  # 자동 GPU 할당
"""
import argparse
import os
import sys
import time
import signal
import threading
from pathlib import Path
import subprocess
import json
from datetime import datetime
from typing import Optional
import yaml

# TSPipe 경로 추가
REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "soft_target"
sys.path.insert(0, str(REPO_ROOT))

from tspipe.failover_logger import init_experiment_logger, finalize_experiment_logger
from tspipe.gpu_health_monitor import GPUHealthMonitor
from gpu_auto_allocator import suggest_gpu_allocation, print_allocation_summary, find_available_gpus


# 이 실험 스크립트에서 사용할 물리 GPU 집합을 고정
PINNED_GPU_SET = [0, 3, 4, 6]
PINNED_GPU_ENV = ",".join(str(g) for g in PINNED_GPU_SET)

class FailoverExperiment:
    """Failover 실험 관리 클래스"""
    
    def __init__(self, args):
        self.args = args
        self.experiment_name = f"failover_{args.experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = None
        self.tspipe_process = None
        self.monitoring_process = None
        self.experiment_start_time = None
        self.failover_ready_event = threading.Event()
        self.worker_seen_event = threading.Event()
        self.restarted = False
        
    def run(self):
        """실험 실행 메인 함수"""
        try:
            print("🚀 GPU Failover 실험 시작")
            print(f"📋 실험 타입: {self.args.experiment_type}")
            print(f"📊 실험 이름: {self.experiment_name}")
            
            # 0. 🧹 시스템 정리 (포트 충돌 방지)
            print("🧹 시스템 정리 중...")
            self._cleanup_previous_processes()
            
            # 1. 사전 검증
            self._validate_environment()
            
            # 2. 실험 로거 초기화
            self._init_logging()
            
            # 3. 실험 시나리오 실행
            if self.args.experiment_type == 'basic':
                self._run_basic_failover_test()
            elif self.args.experiment_type == 'advanced':
                self._run_advanced_failover_test()
            elif self.args.experiment_type == 'profiling_overhead':
                self._run_profiling_overhead_test()
            else:
                raise ValueError(f"Unknown experiment type: {self.args.experiment_type}")
            
            # 4. 결과 분석 및 저장
            self._finalize_experiment()
            
        except KeyboardInterrupt:
            print("\n⚠️ 실험이 사용자에 의해 중단되었습니다.")
            self._cleanup()
        except Exception as e:
            print(f"❌ 실험 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup()
        
    def _validate_environment(self):
        """실험 환경 검증 및 GPU 자동 할당"""
        print("🔍 실험 환경 검증 중...")
        
        # GPU 자동 할당 수행
        if hasattr(self.args, 'auto_gpu') and self.args.auto_gpu:
            print("🤖 자동 GPU 할당 확인 중... (3개 GPU로 검증)")
            allocation = suggest_gpu_allocation(min_gpus_needed=3)
            print_allocation_summary(allocation)
            
            if not allocation['experiment_feasible']:
                print("❌ TSPipe Failover 검증은 최소 3개 GPU가 필요합니다.")
                print("   다른 사용자의 작업이 끝날 때까지 기다려주세요.")
                sys.exit(1)
            
            # 할당된 GPU를 args에 저장
            self.args.gpu_allocation = allocation
            
            # 실험 타입에 따른 GPU 설정 조정
            if allocation['experiment_type'] == 'simulation_only':
                print("⚠️ 현재 시뮬레이션 테스트만 가능합니다.")
                # target-gpu를 available_gpus[0]으로 설정
                if hasattr(self.args, 'target_gpu'):
                    self.args.target_gpu = allocation['recommended_failover_target']
                else:
                    setattr(self.args, 'target_gpu', allocation['recommended_failover_target'])
                    
            elif allocation['experiment_type'] in ['reduced', 'full']:
                # 실험용과 실패용 GPU 분리 설정
                if hasattr(self.args, 'target_gpu'):
                    self.args.target_gpu = allocation['recommended_failover_target']
                else:
                    setattr(self.args, 'target_gpu', allocation['recommended_failover_target'])
            
            print(f"📌 할당된 실험용 GPU: {allocation['recommended_experiment_gpus']}")
            print(f"📌 실패 테스트 대상 GPU: {allocation['recommended_failover_target']}")
        
        # 기본 GPU 개수 확인
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            print(f"✅ 전체 GPU 감지: {gpu_count}개")
            
            # 자동 할당되지 않은 경우 기존 로직 유지
            if not (hasattr(self.args, 'auto_gpu') and self.args.auto_gpu):
                available_gpus = find_available_gpus()
                if len(available_gpus) < 2:
                    print(f"⚠️ 경고: 사용 가능한 GPU가 {len(available_gpus)}개입니다.")
                    print(f"   사용 가능한 GPU: {available_gpus}")
                    
                    response = input("계속 진행하시겠습니까? (y/N): ")
                    if response.lower() != 'y':
                        sys.exit(0)
                    
        except ImportError:
            print("❌ PyTorch를 찾을 수 없습니다.")
            sys.exit(1)
        
        # TSPipe 모듈 확인
        try:
            import tspipe
            print(f"✅ TSPipe 모듈 확인 완료")
        except ImportError:
            print("❌ TSPipe 모듈을 찾을 수 없습니다.")
            sys.exit(1)
            
        # 필수 패키지 확인
        required_packages = ['psutil', 'matplotlib', 'pynvml']
        for pkg in required_packages:
            try:
                __import__(pkg)
                print(f"✅ {pkg} 패키지 확인")
            except ImportError:
                print(f"❌ {pkg} 패키지가 설치되지 않았습니다.")
                sys.exit(1)
        
        print("✅ 환경 검증 완료")
        
    def _init_logging(self):
        """실험 로깅 초기화"""
        self.logger = init_experiment_logger(
            self.experiment_name, 
            output_dir=self.args.output_dir
        )
        self.logger.start_metrics_collection(interval_seconds=self.args.metric_interval)
        
        self.logger.log_message("🚀 Failover 실험 시작", {
            'experiment_type': self.args.experiment_type,
            'args': vars(self.args)
        })
        
    def _run_basic_failover_test(self):
        """기본 Failover 테스트"""
        print("🔬 기본 Failover 테스트 실행")
        
        # 1. TSPipe 시작 (failover 활성화)
        self._start_tspipe_with_failover()
        
        # 2. 정상 동작 확인 (몇 배치 실행)
        self._wait_for_normal_operation(batches=5)
        self._assert_tspipe_alive("초기 정상 동작 단계")

        # 2.5. Failover 시스템/워커 준비 대기
        self._wait_until_failover_ready(timeout_seconds=60)
        
        # 3. GPU 실패 시뮬레이션
        target_gpu = self.args.target_gpu if self.args.target_gpu >= 0 else 4  # 기본값: GPU 4
        self._simulate_gpu_failure(target_gpu)
        
        # 4. 복구 과정 모니터링
        self._monitor_recovery_process()

        # 4.5. hard failover restart config가 생성되면 K-1 재시작 시도
        if self._has_restart_config_available():
            self._restart_from_failover_config_if_available()
        elif self.tspipe_process and self.tspipe_process.poll() is not None and self.tspipe_process.returncode != 0:
            # 레거시 경로 호환: 비정상 종료는 했지만 restart config가 없는 경우
            self._restart_from_failover_config_if_available()
        
        # 5. 복구 후 정상 동작 확인
        self._wait_for_normal_operation(batches=3)
        
    def _run_advanced_failover_test(self):
        """고급 Failover 테스트 (다중 GPU 실패)"""
        print("🔬 고급 Failover 테스트 실행 (다중 GPU 실패)")
        
        # 1. TSPipe 시작
        self._start_tspipe_with_failover()
        
        # 2. 정상 동작 확인
        self._wait_for_normal_operation(batches=3)
        self._assert_tspipe_alive("고급 실험 초기 동작 단계")
        
        # 3. 첫 번째 GPU 실패
        self._simulate_gpu_failure(4)
        time.sleep(10)  # 복구 대기
        
        # 4. 두 번째 GPU 실패 (연쇄 실패 시뮬레이션)
        if self.args.second_failure_gpu >= 0:
            self._simulate_gpu_failure(self.args.second_failure_gpu)
        
        # 5. 최종 복구 모니터링
        self._monitor_recovery_process()
        
    def _run_profiling_overhead_test(self):
        """프로파일링 오버헤드 테스트"""
        print("🔬 프로파일링 오버헤드 테스트")
        
        # 1. 프로파일링 없이 실행
        print("📊 프로파일링 비활성화 상태로 테스트...")
        self._start_tspipe_with_failover(enable_profiling=False)
        baseline_time = self._measure_execution_time(batches=10)
        self._stop_tspipe()
        
        # 2. 프로파일링 활성화하여 실행
        print("📊 프로파일링 활성화 상태로 테스트...")
        self._start_tspipe_with_failover(enable_profiling=True)
        profiling_time = self._measure_execution_time(batches=10)
        self._stop_tspipe()
        
        # 3. 오버헤드 계산 및 기록
        overhead_percent = ((profiling_time - baseline_time) / baseline_time) * 100
        
        self.logger.log_message("📊 프로파일링 오버헤드 측정 완료", {
            'baseline_time_seconds': baseline_time,
            'profiling_time_seconds': profiling_time, 
            'overhead_percent': overhead_percent
        })
        
        print(f"📈 프로파일링 오버헤드: {overhead_percent:.2f}%")
        
    def _start_tspipe_with_failover(self, enable_profiling=True):
        """TSPipe를 failover 기능과 함께 시작"""
        print("🔧 TSPipe를 failover 기능과 함께 시작...")
        
        benchmark_dir = str(BENCHMARK_DIR)
        
        cmd = [
            sys.executable, "train_kd_profiling.py",
            "--img_root=/nas-ssd/datasets/imagenet2012/imagenet",
            "--save_root=./results/failover_test/",
            "--t_model=./results/base/base-i100-vit-large/model_best.pth.tar",
            "--s_init=./results/base/base-i100-resnet152/initial_r152.pth.tar",
            "--kd_mode=st",
            "--lambda_kd=0.1",
            "--t_name=vit_large",
            "--s_name=resnet152",
            "--T=4.0",
            "--data_name=imagenet100",
            "--num_class=100",
            "--batch_size=4",  # 16 -> 4로 축소하여 메모리 부족 방지
            "--tspipe-enable",
            "--tspipe-config=tspipe.yaml", 
            "--num-nodes=1",
            "--rank=0",
            "--ip=localhost",
            "--epochs=1",  # 실험용으로 짧게 설정
            "--max_step_profiling=10",  # 20 -> 10으로 축소
            "--note=failover-experiment",
            "--soft-failover-enable",
            "--soft-failover-auto-restart",
            # Failover 관련 옵션들
            "--failover-enable",
            "--failover-experiment", self.experiment_name,
            "--backup-gpus=5,6",  # GPU 5, 6을 백업으로 사용
            "--health-check-interval=3",
            "--target-fail-gpu", str(self.args.target_gpu),
            "--fail-after-batches=3"  # 3배치 후 빠르게 실패
        ]
        
        if not enable_profiling:
            cmd.append("--disable-profiling")
            
        if self.args.verbose:
            print(f"🔧 TSPipe 명령어: {' '.join(cmd)}")

        # 이 실험에서는 GPU 0,3,4,6만 보이도록 CUDA_VISIBLE_DEVICES를 고정
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = PINNED_GPU_ENV
        if self.args.verbose:
            print(f"🎯 CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

        # 프로세스 시작
        self.experiment_start_time = time.time()
        self.tspipe_process = subprocess.Popen(
            cmd,
            cwd=benchmark_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        # 로그 모니터링 시작
        self._start_log_monitoring()
        
        self.logger.log_message("🔧 TSPipe 프로세스 시작", {
            'pid': self.tspipe_process.pid,
            'command': ' '.join(cmd)
        })
        
    def _start_log_monitoring(self):
        """TSPipe 로그 실시간 모니터링"""
        import threading
        
        def monitor_stdout():
            for line in iter(self.tspipe_process.stdout.readline, ''):
                if line:
                    print(f"[TSPipe] {line.strip()}")
                    if "Failover 시스템 초기화 완료" in line:
                        self.failover_ready_event.set()
                    if "GPU" in line and "Worker" in line:
                        self.worker_seen_event.set()
                    # 로그에서 중요한 이벤트 감지
                    if "GPU" in line and "failure" in line:
                        self.logger.log_message("🚨 GPU 실패 감지 (로그)", {'log_line': line.strip()})
                    if "GPU_FAILURE" in line:
                        self._record_gpu_failure_from_log(line.strip())
                    elif "recovery" in line:
                        self.logger.log_message("🔄 복구 과정 감지 (로그)", {'log_line': line.strip()})
        
        def monitor_stderr():
            for line in iter(self.tspipe_process.stderr.readline, ''):
                if line:
                    print(f"[TSPipe-ERROR] {line.strip()}")
                    
        self.stdout_thread = threading.Thread(target=monitor_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _record_gpu_failure_from_log(self, log_line: str):
        """TSPipe GPU_FAILURE 로그를 failover event로 반영"""
        if not self.logger:
            return

        try:
            payload = {}
            if '|' in log_line:
                json_part = log_line.split('|', 1)[1].strip()
                payload = json.loads(json_part)

            gpu_id = payload.get('gpu_id')
            partition_id = payload.get('partition_id')
            recovery_time_ms = payload.get('recovery_time_ms')
            details = payload.get('details', {})

            self.logger.log_failover_event(
                'gpu_failure',
                gpu_id=gpu_id,
                partition_id=partition_id,
                recovery_time_ms=recovery_time_ms,
                details=details
            )
        except Exception as e:
            self.logger.log_message("⚠️ GPU_FAILURE 로그 파싱 실패", {
                'log_line': log_line,
                'error': str(e)
            })

    def _wait_until_failover_ready(self, timeout_seconds: int = 60):
        """Failover 모니터/워커가 준비될 때까지 대기"""
        print("⏳ Failover 시스템 준비 대기 중...")
        start = time.time()

        while time.time() - start < timeout_seconds:
            self._assert_tspipe_alive("Failover 준비 대기 단계")

            if self.failover_ready_event.is_set():
                # 워커 생성 로그가 없어도 최소한 자식 PID/compute PID가 보이면 진행
                if self._has_targetable_worker_state(self.args.target_gpu):
                    print("✅ Failover/워커 준비 확인 완료")
                    return

            time.sleep(1)

        raise RuntimeError("Failover 시스템 준비 확인 타임아웃: 워커 PID를 찾을 수 없습니다.")

    def _has_targetable_worker_state(self, gpu_id: int) -> bool:
        if not self.tspipe_process:
            return False
        try:
            import psutil
            parent_proc = psutil.Process(self.tspipe_process.pid)
            child_pids = {p.pid for p in parent_proc.children(recursive=True)}
        except Exception:
            return False

        if not child_pids:
            return False

        gpu_pids = self._get_compute_pids_on_gpu(gpu_id)
        if not gpu_pids:
            return False
        return any(pid in child_pids for pid in gpu_pids)
        
    def _wait_for_normal_operation(self, batches=5):
        """정상 동작 확인 (지정된 배치 수만큼 대기)"""
        print(f"⏳ 정상 동작 확인 중... ({batches}배치 대기)")
        
        wait_time = batches * 2  # 배치당 약 2초 예상
        start_time = time.time()
        
        for i in range(wait_time):
            if self.tspipe_process and self.tspipe_process.poll() is not None:
                print("⚠️ TSPipe 프로세스가 예상보다 빨리 종료되었습니다.")
                break
                
            time.sleep(1)
            
            # 진행상황 표시
            if i % 5 == 0:
                elapsed = time.time() - start_time
                print(f"⏳ 진행 중... ({elapsed:.1f}초)")
        
        self.logger.log_message("⏳ 정상 동작 확인 완료", {
            'batches_waited': batches,
            'wait_time_seconds': time.time() - start_time
        })

    def _assert_tspipe_alive(self, stage_name: str):
        """TSPipe 프로세스가 살아있는지 확인"""
        if self.tspipe_process and self.tspipe_process.poll() is None:
            return

        return_code = None
        if self.tspipe_process:
            return_code = self.tspipe_process.returncode

        raise RuntimeError(
            f"TSPipe 프로세스가 {stage_name}에서 비정상 종료되었습니다. "
            f"환경(포트/RPC/NCCL) 이슈를 먼저 해결해야 합니다. returncode={return_code}"
        )
        
    def _simulate_gpu_failure(self, gpu_id):
        """GPU 실패 시뮬레이션 - PID 기반 워커 강제 종료(SIGKILL)"""
        print(f"💥 GPU {gpu_id} 장애 주입 시작 (PID 기반 SIGKILL)")

        try:
            killed_pids = self._kill_gpu_worker_process(gpu_id)
            if not killed_pids:
                raise RuntimeError(f"GPU {gpu_id}에 매핑된 TSPipe 워커 PID를 찾지 못했습니다.")

            self.logger.log_failover_event(
                'gpu_failure_simulation_start',
                gpu_id=gpu_id,
                details={
                    'method': 'pid_based_sigkill',
                    'start_time': datetime.now().isoformat(),
                    'killed_pids': killed_pids,
                    'description': 'OS-level worker kill (SIGKILL)'
                }
            )

            print(f"🎯 GPU {gpu_id} 워커 강제 종료 완료: {killed_pids}")

        except Exception as e:
            print(f"❌ GPU {gpu_id} 실패 시뮬레이션 오류: {e}")
            self.logger.log_message("⚠️ GPU 실패 시뮬레이션 실행 실패", {'error': str(e), 'gpu_id': gpu_id})
            
    def _kill_gpu_worker_process(self, gpu_id):
        """특정 GPU의 TSPipe 워커 프로세스만 직접 강제 종료"""
        try:
            import psutil
            if not self.tspipe_process:
                return []

            parent_pid = self.tspipe_process.pid
            parent_proc = psutil.Process(parent_pid)
            child_pids = {p.pid for p in parent_proc.children(recursive=True)}
            if not child_pids:
                print("⚠️ TSPipe 자식 워커 PID가 아직 없습니다.")
                return []

            gpu_pids = self._get_compute_pids_on_gpu(gpu_id)
            target_pids = sorted(
                pid for pid in gpu_pids
                if pid != parent_pid and pid in child_pids
            )

            if not target_pids:
                # GPU 매핑 정보를 못 얻은 경우에도 워커 다운 시나리오를 위해 자식 워커 1개 종료
                fallback_pid = sorted(child_pids)[0]
                print(f"⚠️ GPU {gpu_id} 매핑 워커를 찾지 못해 fallback 워커 PID {fallback_pid}를 종료합니다.")
                target_pids = [fallback_pid]

            killed = []
            for pid in target_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                    print(f"💀 SIGKILL 전송 완료: PID {pid} (GPU {gpu_id})")
                except OSError as e:
                    print(f"⚠️ PID {pid} 종료 실패: {e}")

            return killed
            
        except Exception as e:
            print(f"⚠️ 워커 프로세스 종료 실패: {e}")
            return []

    def _get_compute_pids_on_gpu(self, gpu_id):
        """nvidia-smi 기준 특정 GPU에 올라간 compute PID 목록 반환"""
        gpu_uuid_map = {}
        map_cmd = [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader"
        ]
        map_result = subprocess.run(map_cmd, capture_output=True, text=True, timeout=5)
        if map_result.returncode != 0:
            return set()

        for line in map_result.stdout.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 2 and parts[0].isdigit():
                gpu_uuid_map[int(parts[0])] = parts[1]

        target_uuid = gpu_uuid_map.get(gpu_id)
        if not target_uuid:
            return set()

        app_cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits"
        ]
        app_result = subprocess.run(app_cmd, capture_output=True, text=True, timeout=5)
        if app_result.returncode != 0:
            return set()

        pids = set()
        for line in app_result.stdout.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] == target_uuid:
                pids.add(int(parts[0]))

        return pids
        
    def _monitor_recovery_process(self):
        """복구 과정 모니터링"""
        print("🔍 복구 과정 모니터링 중...")
        
        recovery_start_time = time.time()
        max_recovery_time = 120  # 최대 2분 대기
        
        while time.time() - recovery_start_time < max_recovery_time:
            # TSPipe 프로세스 상태 확인
            if self.tspipe_process and self.tspipe_process.poll() is not None:
                print("⚠️ TSPipe 프로세스가 종료되었습니다.")
                break
            
            # GPU 상태 주기적 확인
            self._check_gpu_status()
            
            time.sleep(5)
            
        recovery_time = time.time() - recovery_start_time
        
        self.logger.log_message("🔍 복구 과정 모니터링 완료", {
            'recovery_time_seconds': recovery_time,
            'max_wait_time': max_recovery_time
        })

    def _restart_from_failover_config_if_available(self):
        """TSPipe가 실패 종료된 경우 restart_config 기반 K-1 재시작"""
        restart_dir = BENCHMARK_DIR / "results" / "failover_test" / "failover-experiment"
        emergency_path = restart_dir / "emergency_restart_config.json"
        legacy_path = restart_dir / "restart_config.json"

        if emergency_path.exists():
            restart_config_path = emergency_path
        elif legacy_path.exists():
            restart_config_path = legacy_path
        else:
            print("⚠️ emergency_restart_config.json/restart_config.json이 없어 K-1 재시작을 건너뜁니다.")
            return

        try:
            with open(restart_config_path, 'r') as f:
                restart_config = json.load(f)

            failed_gpu = restart_config.get('failed_gpu')
            partition = restart_config.get('partition', {})
            original = restart_config.get('original_config', {})
            original_online = original.get('online', [])
            original_target = original.get('target', [])

            use_partition_payload = (
                isinstance(partition, dict)
                and isinstance(partition.get('snet_partition'), list)
                and isinstance(partition.get('tnet_partition'), list)
                and isinstance(partition.get('gpu_assignment'), list)
            )

            checkpoint_path = restart_config.get('checkpoint_path') or restart_config.get('emergency_checkpoint_path')

            if use_partition_payload:
                # Prefer DP-repartitioned K-1 config generated during hard failover.
                new_online = [int(v) for v in partition.get('snet_partition', [])]
                new_target = [int(v) for v in partition.get('tnet_partition', [])]
                visible_devices = [str(int(g)) for g in partition.get('gpu_assignment', [])]
                config_source = 'partition_payload'
            else:
                # Backward compatibility: derive K-1 split by removing failed GPU index.
                if not original_online or not original_target:
                    print("⚠️ restart_config에 원본 분할 정보가 없어 재시작 불가")
                    return
                if failed_gpu is None or failed_gpu < 0 or failed_gpu >= len(original_online):
                    print("⚠️ failed_gpu 값이 유효하지 않아 재시작 불가")
                    return

                new_online = [v for i, v in enumerate(original_online) if i != failed_gpu]
                new_target = [v for i, v in enumerate(original_target) if i != failed_gpu]
                visible_devices = [str(i) for i in range(len(original_online)) if i != failed_gpu]
                config_source = 'legacy_failed_gpu'

            if not new_online or not new_target:
                print("⚠️ 계산된 재시작 분할이 비어 있어 재시작 불가")
                return
            if len(new_online) != len(new_target):
                print("⚠️ online/target 분할 길이가 달라 재시작 불가")
                return
            if not visible_devices:
                print("⚠️ 사용할 CUDA_VISIBLE_DEVICES가 비어 있어 재시작 불가")
                return

            base_cfg_path = BENCHMARK_DIR / "tspipe.yaml"
            with open(base_cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)

            cfg['tspipe']['model_split']['online'] = new_online
            cfg['tspipe']['model_split']['target'] = new_target
            cfg['tspipe']['hard_failover_restart'] = {
                'source_config': str(restart_config_path),
                'source_type': restart_config.get('restart_type', 'legacy'),
                'config_source': config_source,
                'failed_gpu': failed_gpu,
                'visible_devices': visible_devices,
            }

            restart_cfg_path = BENCHMARK_DIR / "tspipe_restart_kminus1.yaml"
            with open(restart_cfg_path, 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            cuda_visible = ",".join(visible_devices)

            print(
                f"🔄 K-1 재시작 시도: failed_gpu={failed_gpu}, "
                f"source={config_source}, CUDA_VISIBLE_DEVICES={cuda_visible}"
            )

            # 부모가 살아있더라도 워커가 이미 붕괴한 상태일 수 있어 재시작 전에 정리한다.
            self._stop_tspipe()
            self._start_tspipe_with_custom_config(str(restart_cfg_path), cuda_visible, checkpoint_path)
            self.restarted = True

            self.logger.log_message("🔄 K-1 재시작 실행", {
                'failed_gpu': failed_gpu,
                'new_online': new_online,
                'new_target': new_target,
                'cuda_visible_devices': cuda_visible,
                'config_source': config_source,
                'restart_payload_path': str(restart_config_path),
                'restart_config_path': str(restart_cfg_path)
            })

            self._wait_for_normal_operation(batches=3)

        except Exception as e:
            print(f"❌ K-1 재시작 실패: {e}")
            self.logger.log_message("❌ K-1 재시작 실패", {'error': str(e)})

    def _has_restart_config_available(self) -> bool:
        """hard failover restart config 존재 여부 확인"""
        restart_dir = BENCHMARK_DIR / "results" / "failover_test" / "failover-experiment"
        emergency_path = restart_dir / "emergency_restart_config.json"
        legacy_path = restart_dir / "restart_config.json"
        return emergency_path.exists() or legacy_path.exists()

    def _start_tspipe_with_custom_config(self, config_path: str, cuda_visible_devices: str, resume_checkpoint: Optional[str] = None):
        """커스텀 tspipe config/CUDA 환경으로 재시작

        cuda_visible_devices 인자는 restart_config에서 가져온 논리 K-1 GPU 집합이지만,
        실제 물리 GPU 집합은 PINNED_GPU_SET (0,3,4,6)으로 고정한다.
        """
        benchmark_dir = str(BENCHMARK_DIR)
        cmd = [
            sys.executable, "train_kd_profiling.py",
            "--img_root=/nas-ssd/datasets/imagenet2012/imagenet",
            "--save_root=./results/failover_test/",
            "--t_model=./results/base/base-i100-vit-large/model_best.pth.tar",
            "--s_init=./results/base/base-i100-resnet152/initial_r152.pth.tar",
            "--kd_mode=st",
            "--lambda_kd=0.1",
            "--t_name=vit_large",
            "--s_name=resnet152",
            "--T=4.0",
            "--data_name=imagenet100",
            "--num_class=100",
            "--batch_size=4",
            "--tspipe-enable",
            f"--tspipe-config={Path(config_path).name}",
            "--num-nodes=1",
            "--rank=0",
            "--ip=localhost",
            "--epochs=1",
            "--max_step_profiling=10",
            "--note=failover-restart-kminus1",
            "--soft-failover-enable",
            "--soft-failover-auto-restart",
        ]

        if resume_checkpoint:
            cmd.append(f"--resume-checkpoint={resume_checkpoint}")

        env = os.environ.copy()
        # 재시작 시에도 물리 GPU 0,3,4,6만 보이도록 고정
        env['CUDA_VISIBLE_DEVICES'] = PINNED_GPU_ENV

        if self.args.verbose:
            print(f"🔧 K-1 재시작 명령어: {' '.join(cmd)}")
            print(f"🎯 CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

        self.tspipe_process = subprocess.Popen(
            cmd,
            cwd=benchmark_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        self._start_log_monitoring()
        
    def _check_gpu_status(self):
        """GPU 상태 확인"""
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            
            status = {}
            for gpu_id in range(gpu_count):
                try:
                    # GPU 메모리 정보 확인
                    with torch.cuda.device(gpu_id):
                        mem_info = torch.cuda.mem_get_info()
                        status[gpu_id] = {
                            'available': True,
                            'free_memory_mb': mem_info[0] / (1024 * 1024),
                            'total_memory_mb': mem_info[1] / (1024 * 1024)
                        }
                except Exception as e:
                    status[gpu_id] = {
                        'available': False,
                        'error': str(e)
                    }
            
            # 로깅 (너무 자주 하지 않도록 조절)
            if hasattr(self, '_last_gpu_status_log'):
                if time.time() - self._last_gpu_status_log < 30:  # 30초마다만 로깅
                    return
                    
            self._last_gpu_status_log = time.time()
            self.logger.log_message("🔍 GPU 상태 체크", status)
            
        except Exception as e:
            print(f"⚠️ GPU 상태 확인 실패: {e}")
            
    def _measure_execution_time(self, batches=10):
        """실행 시간 측정"""
        start_time = time.time()
        self._wait_for_normal_operation(batches)
        return time.time() - start_time
        
    def _stop_tspipe(self):
        """TSPipe 프로세스 중지"""
        if self.tspipe_process:
            print("🛑 TSPipe 프로세스 중지 중...")
            
            try:
                self.tspipe_process.terminate()
                self.tspipe_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("⚠️ 프로세스가 정상 종료되지 않아 강제 종료합니다.")
                self.tspipe_process.kill()
                self.tspipe_process.wait()
            
            self.logger.log_message("🛑 TSPipe 프로세스 중지 완료")
            self.tspipe_process = None
        
    def _finalize_experiment(self):
        """실험 완료 및 결과 저장"""
        print("📊 실험 결과 분석 및 저장 중...")
        
        if self.experiment_start_time:
            total_time = time.time() - self.experiment_start_time
            self.logger.log_message("🏁 실험 완료", {
                'total_experiment_time_seconds': total_time
            })
        
        # TSPipe 프로세스 정리
        self._stop_tspipe()
        
        # 실패 시뮬레이션 프로세스 정리
        if hasattr(self, 'failure_simulation_process') and self.failure_simulation_process:
            try:
                self.failure_simulation_process.terminate()
                self.failure_simulation_process.wait(timeout=5)
            except:
                pass
        
        # 로거 종료 및 결과 저장
        if self.logger:
            result_dir = finalize_experiment_logger()
            print(f"📁 실험 결과 저장 위치: {result_dir}")
            
            # 결과 요약 출력
            self._print_experiment_summary(result_dir)
        
    def _print_experiment_summary(self, result_dir):
        """실험 결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 실험 결과 요약")
        print("="*60)
        print(f"🏷️ 실험 이름: {self.experiment_name}")
        print(f"📂 결과 디렉토리: {result_dir}")
        print(f"📈 실험 타입: {self.args.experiment_type}")
        
        if result_dir:
            # 주요 파일들 나열
            main_files = [
                "experiment_summary.json",
                "performance.jsonl", 
                "failover_events.jsonl",
                "plots/cpu_performance.png",
                "plots/gpu_memory.png"
            ]
            
            print("\n📁 생성된 주요 파일들:")
            for file in main_files:
                file_path = Path(result_dir) / file
                if file_path.exists():
                    size = file_path.stat().st_size
                    print(f"  ✅ {file} ({size} bytes)")
                else:
                    print(f"  ❌ {file} (생성되지 않음)")
        
        print("\n🎯 교수님께 보고할 자료:")
        print(f"  1. 실험 요약: {result_dir}/experiment_summary.json")
        print(f"  2. 성능 그래프: {result_dir}/plots/")
        print(f"  3. 상세 로그: {result_dir}/experiment.log")
        print("="*60 + "\n")
        
    def _cleanup(self):
        """정리 작업"""
        print("🧹 정리 작업 수행 중...")
        
        # 프로세스들 종료
        self._stop_tspipe()
        
        # 로거 종료
        if self.logger:
            try:
                finalize_experiment_logger()
            except:
                pass
    
    def _cleanup_previous_processes(self):
        """이전 실행의 잔여 프로세스들 정리"""
        try:
            import subprocess
            import sys
            
            # cleanup_processes.py 실행
            cleanup_script = Path(__file__).parent / "cleanup_processes.py"
            if cleanup_script.exists():
                result = subprocess.run(
                    [sys.executable, str(cleanup_script)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    print("✅ 프로세스 정리 완료")
                else:
                    print(f"⚠️ 프로세스 정리 중 일부 오류: {result.stderr}")
            else:
                print("⚠️ cleanup_processes.py 스크립트를 찾을 수 없음")
                
        except subprocess.TimeoutExpired:
            print("⚠️ 프로세스 정리 타임아웃, 계속 진행...")
        except Exception as e:
            print(f"⚠️ 프로세스 정리 중 오류: {e}, 계속 진행...")

        # 레거시 임시 신호 파일 정리
        try:
            for file_name in os.listdir('/tmp'):
                if file_name.startswith('gpu_failure_signal_') and file_name.endswith('.json'):
                    os.remove(os.path.join('/tmp', file_name))
        except Exception:
            pass
        
        # 추가 대기 시간 (포트 해제 보장)
        print("⏳ 포트 해제 대기 (5초)...")
        import time
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="TSPipe GPU Failover 실험")
    
    parser.add_argument(
        "--experiment-type",
        choices=['basic', 'advanced', 'profiling_overhead'],
        default='basic',
        help="실험 타입 선택"
    )
    
    parser.add_argument(
        "--target-gpu",
        type=int,
        default=4,
        help="실패 시뮬레이션 대상 GPU ID (기본값: 4)"
    )
    
    parser.add_argument(
        "--second-failure-gpu",
        type=int,
        default=-1,
        help="고급 실험용: 두 번째 실패 GPU ID (기본값: 없음)"
    )
    
    parser.add_argument(
        "--auto-gpu",
        action="store_true",
        help="자동으로 사용 가능한 GPU를 감지하고 할당"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./failover_results",
        help="실험 결과 저장 디렉토리"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )

    parser.add_argument(
        "--metric-interval",
        type=float,
        default=0.1,
        help="성능 메트릭 수집 주기(초). 논문용 고해상도 캡처는 0.1 권장"
    )
    
    args = parser.parse_args()
    
    # 결과 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 실험 실행
    experiment = FailoverExperiment(args)
    experiment.run()


if __name__ == "__main__":
    main()