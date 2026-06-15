"""Failover 실험 로깅 및 결과 분석 시스템"""
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, asdict
import threading
import psutil
import torch


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터"""
    timestamp: str
    batch_id: int
    iteration_time_ms: float
    gpu_utilization: Dict[int, float]  # GPU ID -> 사용률%
    gpu_memory_used: Dict[int, float]  # GPU ID -> 메모리 사용량 MB
    cpu_usage_percent: float
    total_memory_mb: float
    active_gpus: List[int]
    failed_gpus: List[int]
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class FailoverEvent:
    """Failover 이벤트 기록"""
    timestamp: str
    event_type: str  # "gpu_failure", "recovery_start", "recovery_complete", "repartition"
    gpu_id: Optional[int]
    partition_id: Optional[int]
    old_config: Optional[Dict]
    new_config: Optional[Dict]
    recovery_time_ms: Optional[float]
    details: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


class FailoverExperimentLogger:
    """Failover 실험 전체 로깅 및 분석"""
    
    def __init__(self, experiment_name: str, output_dir: str = "./failover_logs"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.start_time = datetime.now()
        self.experiment_id = f"{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # 로그 디렉토리 생성
        self.log_dir = os.path.join(output_dir, self.experiment_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 로그 파일들
        self.main_log_file = os.path.join(self.log_dir, "experiment.log")
        self.performance_log_file = os.path.join(self.log_dir, "performance.jsonl")
        self.failover_events_file = os.path.join(self.log_dir, "failover_events.jsonl")
        self.summary_file = os.path.join(self.log_dir, "experiment_summary.json")
        
        # 메트릭 수집
        self.performance_metrics: List[PerformanceMetrics] = []
        self.failover_events: List[FailoverEvent] = []
        self.current_config: Dict = {}
        
        # 백그라운드 메트릭 수집
        self.collecting_metrics = False
        self.metrics_thread: Optional[threading.Thread] = None
        
        self._init_experiment_log()
        
    def _init_experiment_log(self):
        """실험 초기 정보 로깅"""
        init_info = {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'start_time': self.start_time.isoformat(),
            'system_info': self._get_system_info(),
            'gpu_info': self._get_gpu_info()
        }
        
        self._write_to_file(self.summary_file, json.dumps(init_info, indent=2))
        self.log_message("📋 Failover 실험 시작", init_info)
        
    def _get_system_info(self) -> Dict:
        """시스템 정보 수집"""
        return {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': str(psutil.sys.version_info),
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }
        
    def _get_gpu_info(self) -> Dict:
        """GPU 정보 수집"""
        gpu_info = {}
        try:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info[i] = {
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                }
        except Exception as e:
            gpu_info = {'error': str(e)}
        return gpu_info
    
    def start_metrics_collection(self, interval_seconds: float = 1.0):
        """백그라운드 성능 메트릭 수집 시작"""
        if self.collecting_metrics:
            return
            
        self.collecting_metrics = True
        self.metrics_thread = threading.Thread(
            target=self._collect_metrics_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self.metrics_thread.start()
        self.log_message("📊 성능 메트릭 수집 시작", {'interval': interval_seconds})
        
    def stop_metrics_collection(self):
        """성능 메트릭 수집 중지"""
        self.collecting_metrics = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        self.log_message("📊 성능 메트릭 수집 중지")
        
    def _collect_metrics_loop(self, interval_seconds: float):
        """메트릭 수집 루프"""
        batch_id = 0
        while self.collecting_metrics:
            try:
                metrics = self._collect_current_metrics(batch_id)
                self.performance_metrics.append(metrics)
                
                # JSON Lines 형식으로 즉시 파일에 저장
                self._append_to_jsonl(self.performance_log_file, metrics.to_dict())
                
                batch_id += 1
                time.sleep(interval_seconds)
            except Exception as e:
                self.log_message("❌ 메트릭 수집 오류", {'error': str(e)})
                time.sleep(interval_seconds)
    
    def _collect_current_metrics(self, batch_id: int) -> PerformanceMetrics:
        """현재 성능 메트릭 수집"""
        timestamp = datetime.now().isoformat()
        
        # GPU 정보 수집
        gpu_utilization = {}
        gpu_memory_used = {}
        active_gpus = []
        failed_gpus = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            for gpu_id in range(torch.cuda.device_count()):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    
                    # GPU 사용률
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization[gpu_id] = util.gpu
                    
                    # GPU 메모리
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used[gpu_id] = mem_info.used / (1024**2)  # MB
                    
                    active_gpus.append(gpu_id)
                    
                except Exception as e:
                    failed_gpus.append(gpu_id)
                    gpu_utilization[gpu_id] = -1
                    gpu_memory_used[gpu_id] = -1
                    
        except ImportError:
            # pynvml이 없는 경우 PyTorch로 대체
            for gpu_id in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(gpu_id):
                        allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                        gpu_memory_used[gpu_id] = allocated
                        gpu_utilization[gpu_id] = 50.0  # 임시값
                        active_gpus.append(gpu_id)
                except Exception:
                    failed_gpus.append(gpu_id)
                    gpu_utilization[gpu_id] = -1
                    gpu_memory_used[gpu_id] = -1
        
        # 시스템 리소스
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            batch_id=batch_id,
            iteration_time_ms=0.0,  # TODO: 실제 iteration time 측정
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            cpu_usage_percent=cpu_usage,
            total_memory_mb=memory_info.used / (1024**2),
            active_gpus=active_gpus,
            failed_gpus=failed_gpus
        )
    
    def log_failover_event(self, event_type: str, **kwargs):
        """Failover 이벤트 로깅"""
        event = FailoverEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            gpu_id=kwargs.get('gpu_id'),
            partition_id=kwargs.get('partition_id'),
            old_config=kwargs.get('old_config'),
            new_config=kwargs.get('new_config'),
            recovery_time_ms=kwargs.get('recovery_time_ms'),
            details=kwargs.get('details', {})
        )
        
        self.failover_events.append(event)
        self._append_to_jsonl(self.failover_events_file, event.to_dict())
        
        # 콘솔에 중요한 이벤트 출력
        emoji_map = {
            'gpu_failure': '💥',
            'recovery_start': '🔄',
            'recovery_complete': '✅',
            'repartition': '🔀',
            'worker_restart': '🚀',
            'profiler_restart': '📊'
        }
        emoji = emoji_map.get(event_type, '📝')
        
        self.log_message(f"{emoji} {event_type.upper()}", event.to_dict())
    
    def log_message(self, message: str, data: Any = None):
        """일반 메시지 로깅"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}"
        
        if data:
            if isinstance(data, dict):
                log_entry += f" | {json.dumps(data, ensure_ascii=False)}"
            else:
                log_entry += f" | {data}"
        
        print(log_entry)  # 콘솔 출력
        
        # 파일 저장
        self._append_to_file(self.main_log_file, log_entry + "\n")
    
    def update_config(self, new_config: Dict):
        """현재 파이프라인 설정 업데이트"""
        old_config = self.current_config.copy()
        self.current_config = new_config.copy()
        
        self.log_message("⚙️ 파이프라인 설정 업데이트", {
            'old_config': old_config,
            'new_config': new_config
        })
    
    def finalize_experiment(self) -> str:
        """실험 종료 및 최종 결과 저장"""
        self.stop_metrics_collection()
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # 최종 요약 생성
        summary = self._generate_experiment_summary(duration)
        
        # 요약 파일 저장
        summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
        self._write_to_file(self.summary_file, summary_json)
        
        # 그래프 생성
        self._generate_plots()
        
        self.log_message("🏁 실험 완료", {
            'duration_seconds': duration,
            'total_events': len(self.failover_events),
            'total_metrics': len(self.performance_metrics),
            'output_directory': self.log_dir
        })
        
        return self.log_dir
    
    def _generate_experiment_summary(self, duration_seconds: float) -> Dict:
        """실험 요약 생성"""
        # Failover 이벤트 분석
        failure_events = [e for e in self.failover_events if e.event_type == 'gpu_failure']
        recovery_events = [e for e in self.failover_events if e.event_type == 'recovery_complete']
        
        # 성능 통계
        if self.performance_metrics:
            cpu_usages = [m.cpu_usage_percent for m in self.performance_metrics]
            avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
        else:
            avg_cpu_usage = 0
        
        return {
            'experiment_info': {
                'name': self.experiment_name,
                'id': self.experiment_id,
                'duration_seconds': duration_seconds,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            },
            'failover_statistics': {
                'total_failures': len(failure_events),
                'total_recoveries': len(recovery_events),
                'avg_recovery_time_ms': self._calculate_avg_recovery_time(),
                'failure_gpu_ids': list(set(e.gpu_id for e in failure_events if e.gpu_id is not None))
            },
            'performance_statistics': {
                'total_metrics_collected': len(self.performance_metrics),
                'avg_cpu_usage_percent': avg_cpu_usage,
                'gpu_utilization_summary': self._analyze_gpu_utilization()
            },
            'file_locations': {
                'main_log': self.main_log_file,
                'performance_data': self.performance_log_file,
                'failover_events': self.failover_events_file,
                'plots_directory': os.path.join(self.log_dir, "plots")
            }
        }
    
    def _calculate_avg_recovery_time(self) -> float:
        """평균 복구 시간 계산"""
        recovery_times = [
            e.recovery_time_ms for e in self.failover_events 
            if e.event_type == 'recovery_complete' and e.recovery_time_ms is not None
        ]
        if recovery_times:
            return sum(recovery_times) / len(recovery_times)
        return 0.0
    
    def _analyze_gpu_utilization(self) -> Dict:
        """GPU 사용률 분석"""
        if not self.performance_metrics:
            return {}
        
        gpu_stats = {}
        for metrics in self.performance_metrics:
            for gpu_id, utilization in metrics.gpu_utilization.items():
                if utilization >= 0:  # 유효한 값만
                    if gpu_id not in gpu_stats:
                        gpu_stats[gpu_id] = []
                    gpu_stats[gpu_id].append(utilization)
        
        summary = {}
        for gpu_id, utilizations in gpu_stats.items():
            summary[gpu_id] = {
                'avg_utilization': sum(utilizations) / len(utilizations),
                'max_utilization': max(utilizations),
                'min_utilization': min(utilizations)
            }
        
        return summary
    
    def _generate_plots(self):
        """실험 결과 시각화"""
        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        if not self.performance_metrics:
            return
        
        # 시간축 데이터 준비
        timestamps = [datetime.fromisoformat(m.timestamp) for m in self.performance_metrics]
        start_time = timestamps[0]
        time_seconds = [(t - start_time).total_seconds() for t in timestamps]
        
        # 1. CPU 사용률 그래프
        plt.figure(figsize=(12, 6))
        cpu_usages = [m.cpu_usage_percent for m in self.performance_metrics]
        plt.plot(time_seconds, cpu_usages, label='CPU Usage %', color='blue')
        
        # Failover 이벤트 표시
        for event in self.failover_events:
            event_time = datetime.fromisoformat(event.timestamp)
            event_seconds = (event_time - start_time).total_seconds()
            
            if event.event_type == 'gpu_failure':
                plt.axvline(x=event_seconds, color='red', linestyle='--', alpha=0.7, label='GPU Failure')
            elif event.event_type == 'recovery_complete':
                plt.axvline(x=event_seconds, color='green', linestyle='--', alpha=0.7, label='Recovery')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU Usage (%)')
        plt.title('System Performance During Failover Experiment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'cpu_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. GPU 메모리 사용률 그래프
        plt.figure(figsize=(12, 8))
        
        # GPU별로 메모리 사용량 추적
        gpu_ids = set()
        for m in self.performance_metrics:
            gpu_ids.update(m.gpu_memory_used.keys())
        
        for gpu_id in sorted(gpu_ids):
            memory_usage = []
            for m in self.performance_metrics:
                usage = m.gpu_memory_used.get(gpu_id, -1)
                memory_usage.append(usage if usage >= 0 else None)
            
            plt.plot(time_seconds, memory_usage, label=f'GPU {gpu_id}', marker='o', markersize=2)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('GPU Memory Usage (MB)')
        plt.title('GPU Memory Usage During Failover Experiment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'gpu_memory.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_message("📈 실험 결과 그래프 생성 완료", {'plots_directory': plots_dir})
    
    def _write_to_file(self, filename: str, content: str):
        """파일에 내용 쓰기"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"파일 쓰기 실패 {filename}: {e}")
    
    def _append_to_file(self, filename: str, content: str):
        """파일에 내용 추가"""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(content)
                f.flush()
        except Exception as e:
            print(f"파일 추가 실패 {filename}: {e}")
    
    def _append_to_jsonl(self, filename: str, data: Dict):
        """JSON Lines 형식으로 데이터 추가"""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                f.flush()
        except Exception as e:
            print(f"JSONL 추가 실패 {filename}: {e}")


# 전역 로거 인스턴스 (실험용)
_experiment_logger: Optional[FailoverExperimentLogger] = None

def get_experiment_logger() -> Optional[FailoverExperimentLogger]:
    """전역 실험 로거 반환"""
    return _experiment_logger

def init_experiment_logger(experiment_name: str, output_dir: str = "./failover_logs") -> FailoverExperimentLogger:
    """실험 로거 초기화"""
    global _experiment_logger
    _experiment_logger = FailoverExperimentLogger(experiment_name, output_dir)
    return _experiment_logger

def finalize_experiment_logger() -> Optional[str]:
    """실험 로거 종료 및 결과 반환"""
    global _experiment_logger
    if _experiment_logger:
        result_dir = _experiment_logger.finalize_experiment()
        _experiment_logger = None
        return result_dir
    return None