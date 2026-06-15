"""GPU 상태 모니터링 및 실패 감지 시스템"""
import subprocess
import threading
import time
import traceback
from datetime import datetime
from typing import Dict, List, Set, Callable, Optional
import torch
import psutil
import os
import signal

from tspipe.logger import Log


class GPUFailureEvent:
    """GPU 실패 이벤트 정보"""
    def __init__(self, gpu_id: int, failure_type: str, timestamp: datetime, error_msg: str = ""):
        self.gpu_id = gpu_id
        self.failure_type = failure_type
        self.timestamp = timestamp
        self.error_msg = error_msg
        self.recovery_timestamp: Optional[datetime] = None
        
    def mark_recovered(self):
        """복구 시점 기록"""
        self.recovery_timestamp = datetime.now()
        
    def get_downtime_seconds(self) -> float:
        """다운타임 계산 (초)"""
        if self.recovery_timestamp:
            return (self.recovery_timestamp - self.timestamp).total_seconds()
        return (datetime.now() - self.timestamp).total_seconds()
        
    def to_dict(self):
        """로깅용 딕셔너리 변환"""
        return {
            'gpu_id': self.gpu_id,
            'failure_type': self.failure_type,
            'timestamp': self.timestamp.isoformat(),
            'error_msg': self.error_msg,
            'recovery_timestamp': self.recovery_timestamp.isoformat() if self.recovery_timestamp else None,
            'downtime_seconds': self.get_downtime_seconds()
        }


class GPUHealthMonitor:
    """GPU 건강 상태 모니터링 및 실패 감지"""
    
    def __init__(self, failure_callback: Callable[[GPUFailureEvent], None], 
                 check_interval: int = 5, experiment_log_file: str = None):
        self.failed_gpus: Set[int] = set()
        self.available_gpus: List[int] = list(range(torch.cuda.device_count()))
        self.failure_callback = failure_callback
        self.check_interval = check_interval
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        self.failure_events: List[GPUFailureEvent] = []
        self.experiment_log_file = experiment_log_file
        
        # GPU별 마지막 성공한 체크 시간 기록
        self.last_success_check: Dict[int, datetime] = {}
        
        # 실험용: 강제로 실패시킬 GPU 목록
        self.force_fail_gpus: Set[int] = set()
        
        Log.i(f"GPUHealthMonitor initialized with {len(self.available_gpus)} GPUs")
        self._log_experiment(f"GPU Health Monitor started with {len(self.available_gpus)} GPUs")
        
    def start_monitoring(self):
        """GPU 상태 모니터링 시작"""
        if self.running:
            Log.w("GPU monitoring is already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        Log.i("GPU health monitoring started")
        self._log_experiment("GPU health monitoring started")
        
    def stop_monitoring(self):
        """GPU 상태 모니터링 중지"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        Log.i("GPU health monitoring stopped")
        self._log_experiment("GPU health monitoring stopped")
        
    def force_gpu_failure(self, gpu_id: int, failure_type: str = "simulation"):
        """실험용: GPU 실패 강제 시뮬레이션"""
        if gpu_id not in self.available_gpus:
            Log.e(f"GPU {gpu_id} is not available for failure simulation")
            return False
            
        self.force_fail_gpus.add(gpu_id)
        Log.w(f"GPU {gpu_id} marked for forced failure ({failure_type})")
        self._log_experiment(f"GPU {gpu_id} marked for forced failure - type: {failure_type}")
        return True
        
    def recover_gpu(self, gpu_id: int):
        """실험용: GPU 복구 시뮬레이션"""
        if gpu_id in self.force_fail_gpus:
            self.force_fail_gpus.remove(gpu_id)
            
        # 해당 GPU의 failure event에 복구 시점 기록
        for event in self.failure_events:
            if event.gpu_id == gpu_id and event.recovery_timestamp is None:
                event.mark_recovered()
                Log.i(f"GPU {gpu_id} recovered after {event.get_downtime_seconds():.2f} seconds")
                self._log_experiment(f"GPU {gpu_id} recovered - downtime: {event.get_downtime_seconds():.2f}s")
                break
                
    def _monitor_loop(self):
        """GPU 상태 주기적 체크 루프"""
        Log.i(f"GPU monitoring loop started (check interval: {self.check_interval}s)")
        
        while self.running:
            try:
                newly_failed = self._check_all_gpus()
                
                # 새로 실패한 GPU들에 대해 콜백 호출
                for gpu_id in newly_failed:
                    failure_event = GPUFailureEvent(
                        gpu_id=gpu_id,
                        failure_type="health_check_failure" if gpu_id not in self.force_fail_gpus else "simulated_failure",
                        timestamp=datetime.now(),
                        error_msg="GPU health check failed or simulated failure"
                    )
                    self.failure_events.append(failure_event)
                    
                    Log.e(f"GPU {gpu_id} failure detected! Type: {failure_event.failure_type}")
                    self._log_experiment(f"GPU {gpu_id} FAILURE DETECTED - {failure_event.failure_type}")
                    
                    # 실패 콜백 호출
                    try:
                        self.failure_callback(failure_event)
                    except Exception as e:
                        Log.e(f"Error in failure callback for GPU {gpu_id}: {e}")
                        Log.e(traceback.format_exc())
                        
                time.sleep(self.check_interval)
                
            except Exception as e:
                Log.e(f"Error in GPU monitoring loop: {e}")
                Log.e(traceback.format_exc())
                time.sleep(self.check_interval)
                
    def _check_all_gpus(self) -> Set[int]:
        """모든 GPU 상태 체크, 새로 실패한 GPU ID 반환"""
        newly_failed = set()
        
        for gpu_id in self.available_gpus:
            if gpu_id in self.failed_gpus:
                continue  # 이미 실패로 표시된 GPU는 스킵
                
            is_healthy = self._is_gpu_healthy(gpu_id)
            
            if not is_healthy:
                newly_failed.add(gpu_id)
                self.failed_gpus.add(gpu_id)
            else:
                self.last_success_check[gpu_id] = datetime.now()
                
        return newly_failed
    
    def _is_gpu_healthy(self, gpu_id: int) -> bool:
        """개별 GPU 건강 상태 체크"""
        try:
            # 강제 실패 시뮬레이션 체크
            if gpu_id in self.force_fail_gpus:
                return False
                
            # 기본 CUDA 컨텍스트 체크
            if not self._check_cuda_context(gpu_id):
                return False
                
            # 메모리 할당 테스트
            if not self._check_memory_allocation(gpu_id):
                return False
                
            # nvidia-smi 응답 체크
            if not self._check_nvidia_smi(gpu_id):
                return False
                
            return True
            
        except Exception as e:
            Log.e(f"GPU {gpu_id} health check failed with exception: {e}")
            return False
    
    def _check_cuda_context(self, gpu_id: int) -> bool:
        """CUDA 컨텍스트 접근 테스트"""
        try:
            with torch.cuda.device(gpu_id):
                # 간단한 연산 수행
                current_device = torch.cuda.current_device()
                return current_device == gpu_id
        except Exception as e:
            Log.d(f"GPU {gpu_id} CUDA context check failed: {e}")
            return False
    
    def _check_memory_allocation(self, gpu_id: int) -> bool:
        """GPU 메모리 할당/해제 테스트"""
        try:
            with torch.cuda.device(gpu_id):
                # 작은 텐서 생성 및 삭제
                test_tensor = torch.zeros(100, device=f'cuda:{gpu_id}')
                del test_tensor
                torch.cuda.empty_cache()
                return True
        except Exception as e:
            Log.d(f"GPU {gpu_id} memory allocation check failed: {e}")
            return False
    
    def _check_nvidia_smi(self, gpu_id: int) -> bool:
        """nvidia-smi 명령어로 GPU 상태 체크"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except Exception as e:
            Log.d(f"GPU {gpu_id} nvidia-smi check failed: {e}")
            return False
    
    def get_healthy_gpus(self) -> List[int]:
        """현재 건강한 GPU 목록 반환"""
        return [gpu_id for gpu_id in self.available_gpus if gpu_id not in self.failed_gpus]
    
    def get_failure_summary(self) -> Dict:
        """실패 요약 정보 반환"""
        summary = {
            'total_gpus': len(self.available_gpus),
            'healthy_gpus': len(self.get_healthy_gpus()),
            'failed_gpus': len(self.failed_gpus),
            'failure_events': [event.to_dict() for event in self.failure_events],
            'current_failed_gpu_ids': list(self.failed_gpus),
            'monitoring_duration_seconds': 0  # TODO: 계산 로직 추가
        }
        return summary
    
    def _log_experiment(self, message: str):
        """실험 로그 기록"""
        timestamp = datetime.now().isoformat()
        log_message = f"[{timestamp}] [GPU_MONITOR] {message}\n"
        
        # 콘솔 출력
        print(f"🔍 {log_message.strip()}")
        
        # 파일 로그 (지정된 경우)
        if self.experiment_log_file:
            try:
                with open(self.experiment_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message)
                    f.flush()
            except Exception as e:
                Log.e(f"Failed to write experiment log: {e}")


class ProcessHealthMonitor:
    """GPU 워커 프로세스 상태 모니터링"""
    
    def __init__(self, process_failure_callback: Callable[[int, str], None]):
        self.monitored_processes: Dict[int, psutil.Process] = {}  # partition_id -> Process
        self.process_failure_callback = process_failure_callback
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def add_process(self, partition_id: int, process: psutil.Process):
        """모니터링할 프로세스 추가"""
        self.monitored_processes[partition_id] = process
        Log.i(f"Added process monitoring for partition {partition_id}, PID: {process.pid}")
        
    def start_monitoring(self):
        """프로세스 모니터링 시작"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        self.monitor_thread.start()
        Log.i("Process health monitoring started")
        
    def stop_monitoring(self):
        """프로세스 모니터링 중지"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
    def _monitor_processes(self):
        """프로세스 상태 주기적 체크"""
        while self.running:
            dead_partitions = []
            
            for partition_id, process in list(self.monitored_processes.items()):
                try:
                    process_dead = not process.is_running()
                    if not process_dead:
                        try:
                            process_dead = process.status() == psutil.STATUS_ZOMBIE
                        except Exception:
                            pass

                    if process_dead:
                        dead_partitions.append(partition_id)
                        Log.e(f"Process for partition {partition_id} (PID: {process.pid}) is not running")
                except Exception as e:
                    dead_partitions.append(partition_id)
                    Log.e(f"Error checking process for partition {partition_id}: {e}")
            
            # 죽은 프로세스들에 대해 콜백 호출
            for partition_id in dead_partitions:
                if not self.running:
                    break
                try:
                    self.process_failure_callback(partition_id, "process_died")
                    del self.monitored_processes[partition_id]
                except Exception as e:
                    Log.e(f"Error in process failure callback for partition {partition_id}: {e}")
            
            time.sleep(5)  # 5초마다 체크
