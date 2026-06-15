import warnings
from argparse import ArgumentParser, Namespace
from collections import OrderedDict, defaultdict
from copy import deepcopy
from enum import Enum
from itertools import chain
from platform import python_version
from queue import Empty
from threading import Thread
from time import sleep, time
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import json

import torch
import yaml
import sys
import psutil
import os
import logging

import tspipe
from tspipe import batch_ops
from tspipe.affinity_manager import AffinityManager
from tspipe.batch_ops import (Batch, BatchQueue, ScatterGatherFn,
                              defaultScatterGatherFn)
from tspipe.batchnorm import DeferredBatchNorm
from tspipe import communicator
from tspipe.communicator import (Channel, Communicator, CommunicatorParam,
                                 DistributedQueue)
from tspipe.gpu_context import StreamDescriptor, StreamType, TaskContext
from tspipe.gpu_task import GpuTask, TaskType
from tspipe.gpu_worker import GpuWorker, LocalWorker
from tspipe.logger import Log
from tspipe.profiler import Operation, ProfilerDelegateWorker, profile_init, TSPipeProfiler
from tspipe.scheduler import TSPipeScheduler
from tspipe.prototype_scheduler import TSPipeSchedulerKD
from tspipe.utils import get_shape, verify_module

# Failover 관련 임포트 추가
from tspipe.gpu_health_monitor import GPUHealthMonitor, GPUFailureEvent, ProcessHealthMonitor
from tspipe.failover_logger import (FailoverExperimentLogger, init_experiment_logger, 
                                  get_experiment_logger, finalize_experiment_logger)

BATCH_FEED_THESH = 4

Tensors = Tuple[torch.Tensor, ...]
TensorOrTensors = Union[torch.Tensor, Tensors]

class BalanceError(ValueError):
    pass

class PipelineRunState(Enum):
    STOPPED = 0
    RUNNING = 1
    PENDING_STOP = 2


class TSPipeMode(Enum):
    SELF_SUPERVISED_MOMENTUM = 0
    SUPERVISED_MOMENTUM = 1

class TSPipe():
    def __init__(self,
                 module_online: torch.nn.Sequential, # student model
                 module_target: torch.nn.Sequential, # teacher model
                 module_predictor: Optional[torch.nn.Sequential], # predictor model <- used for self-supervised learning
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: Callable,
                 target_update_fn: Optional[Callable],
                 momentum: float,
                 artifact_dir: str = '',
                 tspipe_mode: TSPipeMode = TSPipeMode.SELF_SUPERVISED_MOMENTUM,
                 target_train_mode: bool = True,
                 extra_args: Namespace = Namespace(),
                 scatter_gather_fn: ScatterGatherFn = defaultScatterGatherFn,
                ):
        parser = ArgumentParser()
        parser.add_argument("--tspipe-config", required=True, type=str)
        parser.add_argument("--ip", required=True, type=str)
        parser.add_argument("--rank", required=True, type=int, default=0)
        parser.add_argument("--num-nodes", required=True, type=int, default=1)
        
        # Failover 관련 인자들 추가
        parser.add_argument("--failover-enable", action="store_true", help="Enable GPU failover system")
        parser.add_argument("--failover-experiment", type=str, default="", help="Failover experiment name")
        parser.add_argument("--backup-gpus", type=str, default="", help="Comma-separated backup GPU IDs")
        parser.add_argument("--health-check-interval", type=int, default=5, help="GPU health check interval (seconds)")
        parser.add_argument("--target-fail-gpu", type=int, default=-1, help="Target GPU to simulate failure")
        parser.add_argument("--fail-after-batches", type=int, default=10, help="Simulate failure after N batches")
        parser.add_argument("--healthy-checkpoint-interval", type=int, default=20,
                    help="Periodic healthy checkpoint interval in steps")
        parser.add_argument("--checkpoint-benchmark-enable", action="store_true",
                    help="Enable checkpoint benchmark metrics logging (Experiment 0)")
        parser.add_argument("--checkpoint-benchmark-prefix", type=str, default="exp0_checkpoint",
                    help="Prefix for Experiment 0 metrics files")

        args, _ = parser.parse_known_args()
        self.args               = args
        self.module_online      = module_online
        self.module_target      = module_target
        self.module_predictor   = module_predictor
        self.optimizer          = optimizer
        self.loss_fn            = loss_fn
        self.target_update_fn   = target_update_fn
        self.momentum           = momentum
        self.tspipe_mode        = tspipe_mode
        self.target_train_mode  = target_train_mode
        self.extra_args         = extra_args
        self.scatter_gather_fn  = scatter_gather_fn
        
        # Failover 시스템 초기화
        self.failover_enabled = args.failover_enable
        # 백업 GPU는 K-1 재구성 방식으로 통일하기 위해 미사용
        # self.backup_gpu_ids = []
        
        self.gpu_health_monitor: Optional[GPUHealthMonitor] = None
        self.process_health_monitor: Optional[ProcessHealthMonitor] = None
        self.experiment_logger: Optional[FailoverExperimentLogger] = None
        self.failed_workers: Dict[int, GpuWorker] = {}
        self.original_partition_config = None
        self.current_partition_config = None
        self.failover_in_progress = False
        
        # 실험용 실패 시뮬레이션  
        self.target_fail_gpu = args.target_fail_gpu
        self.fail_after_batches = args.fail_after_batches
        self.batch_count = 0
        self.failure_simulated = False
        self.last_healthy_checkpoint_step = 0
        self._emergency_shutdown_started = False
        self.healthy_checkpoint_interval = max(0, args.healthy_checkpoint_interval)
        self.checkpoint_benchmark_enabled = args.checkpoint_benchmark_enable
        self.periodic_checkpoint_enabled = self.failover_enabled or self.checkpoint_benchmark_enabled

        self.last_completed_step_time_ms: Optional[float] = None
        self._pending_save_spike: Optional[Dict] = None
        self._save_event_seq = 0
        benchmark_prefix = args.checkpoint_benchmark_prefix.strip() if args.checkpoint_benchmark_prefix else "exp0_checkpoint"
        artifact_dir = artifact_dir if artifact_dir else '.'
        os.makedirs(artifact_dir, exist_ok=True)
        self.step_metrics_file = f"{artifact_dir}/{benchmark_prefix}_step_metrics.jsonl"
        self.save_events_file = f"{artifact_dir}/{benchmark_prefix}_save_events.jsonl"
        
        # 스케줄러 중단 플래그
        self._shutdown_scheduler = False
        
        self.terminate_futures = [] # futures for termination tasks

        Log.i(f"====== TSPipe (v{tspipe.__version__}) Initializing =====")
        Log.i(f"Running on Python {python_version()}, PyTorch {torch.__version__}")
        # 🔄 Restart 시 새로운 포트값을 읽도록: 매번 함수 호출 (모듈 초기화 블록 우회)
        self._nccl_port, self._rpc_port = communicator._get_nccl_rpc_ports()
        Log.i(f"Fixed ports: NCCL={self._nccl_port}, RPC={self._rpc_port} (re-read from environment)")
        with open(args.tspipe_config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)['tspipe']
        self.config['__artifact_dir'] = artifact_dir

        self.num_devices = torch.cuda.device_count()
        Log.v(f"Detected {self.num_devices}")

        self.rank, self.num_nodes, self.total_world_size = args.rank, args.num_nodes, self.num_devices*args.num_nodes + 1
        self.num_total_devices = self.num_devices*self.num_nodes
        Log.v(f"Starting Node with rank {args.rank}")
        
        # configure number of microbatches
        # assuming all nodes have the same number of GPUs
        Log.v(f"Using {self.num_nodes} nodes x {self.num_devices} GPUs = {self.num_nodes*self.num_devices}")
        Log.v(self.config)
        
        # self.max_step_profiling = self.config['train']['max_step_profiling']
        self.max_step_profiling = None
        
        partition_count = len(self.config['model_split']['online'])
        if len(self.config['model_split']['target']) != partition_count:
            raise ValueError(
                "Online/target partition counts must match: "
                f"online={partition_count}, target={len(self.config['model_split']['target'])}"
            )

        detected_gpu_count = self.num_nodes * self.num_devices
        if detected_gpu_count != partition_count:
            # Single-node launcher restarts can leave stale restart config/GPU visibility.
            # Try to auto-heal rather than crashing immediately.
            if self.num_nodes != 1:
                raise ValueError(
                    "The number of GPUs and the number of partitions must match: "
                    f"gpus={detected_gpu_count}, partitions={partition_count}"
                )

            if detected_gpu_count <= 0:
                raise ValueError(
                    "No visible CUDA devices detected while partitions are configured: "
                    f"partitions={partition_count}"
                )

            if detected_gpu_count < partition_count:
                online_split = [int(v) for v in self.config['model_split']['online']]
                target_split = [int(v) for v in self.config['model_split']['target']]
                while len(online_split) > detected_gpu_count:
                    online_split[-2] += online_split[-1]
                    target_split[-2] += target_split[-1]
                    online_split.pop()
                    target_split.pop()
                self.config['model_split']['online'] = online_split
                self.config['model_split']['target'] = target_split
                Log.w(
                    "⚠️ GPU/partition mismatch auto-fixed by merging tail partitions: "
                    f"gpus={detected_gpu_count}, original_partitions={partition_count}, "
                    f"new_online={online_split}, new_target={target_split}"
                )
            else:
                # More visible GPUs than partitions: use only as many GPUs as partitions.
                self.num_devices = partition_count
                self.total_world_size = self.num_devices * self.num_nodes + 1
                self.num_total_devices = self.num_devices * self.num_nodes
                Log.w(
                    "⚠️ GPU/partition mismatch auto-fixed by limiting visible partitions to configured split: "
                    f"gpus={detected_gpu_count}, partitions={partition_count}, using_gpus={self.num_devices}"
                )
        
        if self.config['gpipe_emulation']['enabled']:
            self.num_ubatches = self.config['gpipe_emulation']['num_ubatch']
            self.num_bwd_ubatches = self.num_ubatches
        else:
            self.num_ubatches = self.num_total_devices - 1
            self.num_bwd_ubatches = self.num_ubatches * 2

        if self.target_update_fn is None:
            warnings.warn("Target update function is not designated. Target network will not be updated.")
        
        # Failover 시스템 초기화
        if self.failover_enabled:
            self._init_failover_system()
            
        # start gpu worker nodes
        self.gpu_workers: List[GpuWorker] = []
        for partition_id in range(self.rank * self.num_devices, (self.rank + 1) * self.num_devices):
            Log.v(f"Spawning worker with partition {partition_id}")
            worker = self.spawn_new_gpu_workers(partition_id)
            self.gpu_workers.append(worker)
            
            # 프로세스 건강 모니터링에 추가
            if self.failover_enabled and self.process_health_monitor:
                import psutil
                try:
                    process = psutil.Process(worker.process.pid)
                    self.process_health_monitor.add_process(partition_id, process)
                except Exception as e:
                    Log.e(f"Failed to add process monitoring for partition {partition_id}: {e}")

        # Failover 모니터링은 학습 시작 전에 즉시 활성화해야 워커 다운을 실시간 감지할 수 있음
        if self.rank == 0 and self.failover_enabled:
            self._start_failover_monitoring()

        # start pipeline for primary node (rank 0)
        if self.rank == 0:
            sleep(1)
            self.start_primary()
        else:
            self.join_workers()

    def _append_jsonl(self, file_path: str, payload: Dict):
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            Log.e(f"Failed to append jsonl to {file_path}: {e}")

    def _build_resume_compatible_checkpoint_payload(self, checkpoint_kind: str) -> Dict:
        """Build a checkpoint payload that train_kd.py's soft resume loader can restore."""
        payload = {
            'checkpoint_kind': str(checkpoint_kind),
            'save_timestamp': time(),
            'batch_count': int(self.batch_count),
            'global_step': int(self.batch_count),
        }

        if self.module_online is not None:
            online_state = self.module_online.state_dict()
            payload['student_state_dict'] = online_state
            # Keep the legacy key for backward compatibility with older tooling.
            payload['model_state_dict'] = online_state

        if self.module_target is not None:
            payload['teacher_state_dict'] = self.module_target.state_dict()

        if self.optimizer is not None:
            try:
                payload['optimizer_state_dict'] = self.optimizer.state_dict()
            except Exception as e:
                Log.w(f"⚠️ Failed to serialize optimizer state for {checkpoint_kind} checkpoint: {e}")

        return payload

    def _record_step_metric(self, step_id: int, step_time_ms: float):
        payload = {
            'event_type': 'step_timing',
            'timestamp_sec': time(),
            'step_id': step_id,
            'step_time_ms': step_time_ms,
            'batch_count': self.batch_count,
            'checkpoint_interval': self.healthy_checkpoint_interval
        }
        self._append_jsonl(self.step_metrics_file, payload)

        if self._pending_save_spike is not None:
            pre_ms = self._pending_save_spike.get('pre_step_time_ms')
            spike_payload = {
                'event_type': 'checkpoint_spike_observed',
                'timestamp_sec': time(),
                'save_event_id': self._pending_save_spike.get('save_event_id'),
                'save_batch_count': self._pending_save_spike.get('batch_count'),
                'pre_step_time_ms': pre_ms,
                'post_step_time_ms': step_time_ms,
                'delta_ms': (step_time_ms - pre_ms) if pre_ms is not None else None,
                'ratio_post_over_pre': (step_time_ms / pre_ms) if pre_ms not in (None, 0) else None,
                'post_step_id': step_id,
            }
            self._append_jsonl(self.save_events_file, spike_payload)
            self._pending_save_spike = None
        
    def start_primary(self):
        """Initialize and start pipeline for the primary node."""
        self.running: PipelineRunState = PipelineRunState.STOPPED        
        self.batch_id: int = 1
        self.highest_scheduled_batch_id: int = 0
        self.forward_complete_batch_idx: int = 0
        self.batch_dict: Dict[int, Tuple[List[Batch], List[Batch],List[Batch], List[Batch]]] = {}
        self.comm = Communicator(None, CommunicatorParam(self.args.ip, self.total_world_size, rank=self.total_world_size-1,
                                                              nccl_port=self._nccl_port, rpc_port=self._rpc_port))
        self.batch_q = BatchQueue()
        self.scheduled_momentum_update = {}
        self.scheduled_lr_update = {}
        
        Log.v("GPU-CPU affinity map:", AffinityManager().get_gpu_affinity_map())
        AffinityManager().set_affinity_for_scheduler()

        # module validity check
        if not self.config['model_split']['online'] or not self.config['model_split']['target']:
            raise NotImplementedError("Model Split is not configured.")
        verify_module(self.module_online)
        verify_module(self.module_target)
        verify_module(self.module_predictor)

        # convert module if deferred batch norm is enabled
        if self.config['train']['deferred_batch_norm']:
            self.module_online = DeferredBatchNorm.convert_deferred_batch_norm(self.module_online, self.num_ubatches)
            self.module_target = DeferredBatchNorm.convert_deferred_batch_norm(self.module_target, self.num_ubatches)
            if self.module_predictor:
                self.module_predictor = DeferredBatchNorm.convert_deferred_batch_norm(self.module_predictor, self.num_ubatches)

        # split module
        self.partitions_online: List[torch.nn.Sequential] = []
        self.partitions_target: List[torch.nn.Sequential] = []
        self.partitions_online, self.partitions_target = \
            self.split_module(self.config['model_split']['online'], self.config['model_split']['target'])
        assert(self.partitions_online)
        assert(self.partitions_target)
        assert(len(self.partitions_online) == len(self.partitions_target))
        self.num_partitions = len(self.partitions_online)
        self.task_scheduler = TSPipeScheduler(self.config, self.batch_q, self.num_partitions)

        # build optimizer map
        param_partitition_map = {}  # self.partitions_online: torch.nn.Sequential로 감싼 모듈 파티션 리스트. 를 딕셔너리로 변환
        for partition_id, partition in enumerate(self.partitions_online):
            for param_id, param in enumerate(partition.parameters()):
                param_partitition_map[param] = param_id, partition_id
        optimizer_pg_param_map = [defaultdict(set) for _ in range(self.num_partitions)]
        for pg_id, pg in enumerate(self.optimizer.param_groups): # optimizer가 관리하는 parameter group 리스트
            for param in pg['params']:
                param_id, partition_id = param_partitition_map[param]
                optimizer_pg_param_map[partition_id][pg_id].add(param_id) #  각 파티션 별로 어떤 parameter group이 있고, 거기에 어떤 parameter ID들이 있는지
        optimizer_pg_param_map = [{k: list(v) for k, v in itm.items()} for itm in optimizer_pg_param_map]
        del param_partitition_map
            
        # enable training mode
        for partition in self.partitions_online:
            partition.train()

        if self.target_train_mode:
            for partition in self.partitions_target:
                partition.train()
        else:
            for partition in self.partitions_target:
                partition.eval()
            warnings.warn("Target train mode is disabled. Target network will be evaluated with eval mode.")        

        # Initialize channels
        for i in range(0, self.num_partitions):
            for prefix in ['task', 'init_config', 'init_model']:
                self.comm.create_channel(f'{prefix}_{i}', i)

        self.comm.create_channel('loss_out', self.num_partitions - 1)
        self.forward_out_queue = DistributedQueue(self.num_partitions - 1, self.comm.rank, 'loss_out')
        
        # collect model after training and profiler logs during training
        self.model_out_channels: List[Channel] = []
        log_channels: List[Channel] = []
        for i in range(0, self.num_partitions):
            self.model_out_channels.append(self.comm.create_channel(f'model_out_{i}', self.num_partitions - 1))
            log_channels.append(self.comm.create_channel(f'log_{i}', self.num_partitions - 1))
        self.comm.create_channel_mux('log', *log_channels)
        

        # barrier to wait other workers
        self.comm.wait_ready()

        # initialize local worker
        lst_local_context = TaskContext.create_local_contexts(self.num_partitions, self.num_ubatches, self.num_bwd_ubatches, self.comm, self.config)
        self.local_worker = LocalWorker(lst_local_context)
        
        for i in range(0, self.num_partitions):
            self.comm.send(f'init_config_{i}', self.args)
            self.comm.send(f'init_config_{i}', self.config)
            self.comm.send(f'init_config_{i}', self.extra_args)
            self.comm.send(f'init_config_{i}', optimizer_pg_param_map[i])
            self.comm.send(f'init_model_{i}', (self.partitions_online[i], self.partitions_target[i]))

        # start profiler
        # with TSPipeProfiler("log.csv"):
        #     profile_init()
        # self.profiler_delegate_worker = ProfilerDelegateWorker(DistributedQueue(None, self.comm.rank, 'log'))

        # start pipeline
        self.start_pipeline()

    def _init_failover_system(self):
        """Failover 시스템 초기화"""
        Log.i("🚨 Initializing GPU failover system...")
        
        # 실험 로거 초기화
        experiment_name = self.args.failover_experiment or f"failover_experiment_{self.rank}"
        self.experiment_logger = init_experiment_logger(experiment_name)
        self.experiment_logger.start_metrics_collection(interval_seconds=2.0)
        
        # 원본 파티션 설정 저장
        self.original_partition_config = {
            'online': self.config['model_split']['online'].copy(),
            'target': self.config['model_split']['target'].copy()
        }
        self.current_partition_config = deepcopy(self.original_partition_config)
        
        # GPU 헬스 모니터 초기화
        experiment_log_file = self.experiment_logger.main_log_file if self.experiment_logger else None
        self.gpu_health_monitor = GPUHealthMonitor(
            failure_callback=self._on_gpu_failure,
            check_interval=self.args.health_check_interval,
            experiment_log_file=experiment_log_file
        )
        
        # 프로세스 헬스 모니터 초기화
        self.process_health_monitor = ProcessHealthMonitor(
            process_failure_callback=self._on_process_failure
        )
        
        Log.i("✅ Failover system initialized")
        if self.experiment_logger:
                self.experiment_logger.log_message("✅ Failover 시스템 초기화 완료 (K-1 재구성 방식)", {
                'original_config': self.original_partition_config
            })

    def _start_failover_monitoring(self):
        """Failover 모니터링 시작"""
        if not self.failover_enabled:
            return
            
        if self.gpu_health_monitor:
            self.gpu_health_monitor.start_monitoring()
            
        if self.process_health_monitor:
            self.process_health_monitor.start_monitoring()
            
        # 실험용 실패 시뮬레이션 스케줄링
        if self.target_fail_gpu >= 0 and self.fail_after_batches > 0:
            Log.w(f"💣 Scheduled failure simulation: GPU {self.target_fail_gpu} after {self.fail_after_batches} batches")
            if self.experiment_logger:
                self.experiment_logger.log_message("💣 실패 시뮬레이션 예약", {
                    'target_gpu': self.target_fail_gpu,
                    'fail_after_batches': self.fail_after_batches
                })

    def _on_gpu_failure(self, failure_event: GPUFailureEvent):
        """GPU 실패 이벤트 핸들러"""
        Log.e(f"💥 GPU {failure_event.gpu_id} failure detected! Type: {failure_event.failure_type}")
        
        if self.experiment_logger:
            self.experiment_logger.log_failover_event(
                'gpu_failure',
                gpu_id=failure_event.gpu_id,
                details={
                    'failure_type': failure_event.failure_type,
                    'timestamp': failure_event.timestamp.isoformat(),
                    'error_msg': failure_event.error_msg
                }
            )
        
        # 🔥 CRITICAL: 장애 감지 즉시 전체 작업 강제 종료
        Log.e(f"🛑 EMERGENCY SHUTDOWN: Terminating all processes due to GPU {failure_event.gpu_id} failure")
        self._emergency_shutdown_and_failover(failure_event.gpu_id)

    def _emergency_shutdown_and_failover(self, failed_gpu_id: int):
        """긴급 종료 및 failover 프로세스 - 사용자 연구 목표에 맞는 구현"""
        if self._emergency_shutdown_started:
            Log.w("⚠️ Emergency shutdown already in progress, skipping duplicate trigger")
            return
        self._emergency_shutdown_started = True

        Log.e("🚨 Starting emergency shutdown and failover process...")

        # Stop background monitors first so they do not keep cascading duplicate
        # callbacks while the emergency shutdown is already in progress.
        if self.gpu_health_monitor is not None:
            self.gpu_health_monitor.running = False
        if self.process_health_monitor is not None:
            self.process_health_monitor.running = False
        if self.experiment_logger is not None:
            self.experiment_logger.collecting_metrics = False
        
        # 1. 현재 학습 루프 강제 중단
        self._force_stop_current_training()
        
        # 2. 실패한 GPU worker 즉시 종료
        self._terminate_failed_workers(failed_gpu_id)

        # 3. C++/RPC abort 이전에 external supervisor가 읽을 restart config를 먼저 남긴다.
        self._prepare_k_minus_1_restart(failed_gpu_id)
        
        # 4. 분산 통신 환경 완전 정리 (Clean Shutdown)
        self._cleanup_distributed_environment()
        
        # 5. 체크포인트 저장 (현재 상태 백업)
        self._save_emergency_checkpoint()
        
        # 6. 프로세스 완전 종료 (재시작은 외부 스크립트가 담당)
        self._terminate_process_for_restart(exit_code=42)

    def _terminate_process_for_restart(self, exit_code: int = 42):
        """Terminate the entire Python process so the external launcher can restart it.

        Hard-failure callbacks are often invoked from background monitor threads.
        `sys.exit()` only stops the calling thread in that case, so the launcher
        never sees the failover exit code. We flush logs explicitly, then force
        a process-wide exit with the restart code.
        """
        Log.e(
            f"💀 Process termination scheduled - external restart required "
            f"(exit_code={int(exit_code)})"
        )
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        try:
            logging.shutdown()
        except Exception:
            pass
        os._exit(int(exit_code))

    def _cleanup_distributed_environment(self):
        """분산 통신 환경 완전 정리 (Clean Shutdown)"""
        try:
            Log.e("🧹 Cleaning up distributed environment...")
            
            import torch.distributed as dist
            import torch.distributed.rpc as rpc
            
            # 1. RPC 종료 (5초 타임아웃으로 충분한 시간 제공)
            if rpc.is_available():
                try:
                    rpc.shutdown(timeout=5.0)  # RPC 스레드 풀 완전 정리 대기
                except Exception:
                    pass  # timeout/error 무시
            
            # 2. NCCL/Gloo 프로세스 그룹 종료
            if dist.is_initialized():
                try:
                    Log.e("🔌 Destroying process groups...")
                    dist.destroy_process_group() 
                    Log.e("✅ Process groups destroyed")
                except Exception as e:
                    Log.e(f"⚠️ Process group destruction error: {e}")
            
            # 3. CUDA 컨텍스트 정리 
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    Log.e("✅ CUDA contexts cleaned")
            except Exception as e:
                Log.e(f"⚠️ CUDA cleanup error: {e}")
                Log.e("🧹 Distributed environment cleanup completed")
            
            
        except Exception as e:
            Log.e(f"❌ Distributed cleanup failed: {e}")
        
    def _force_stop_current_training(self):
        """현재 진행 중인 학습 루프 강제 중단"""
        Log.w("⏹️ Forcing stop of current training loop...")
        
        # 스케줄러 스레드 중단
        if hasattr(self, 'thread') and self.thread.is_alive():
            # 스케줄링 중단 신호
            self._shutdown_scheduler = True
            
        # 모든 GPU worker에게 즉시 중단 신호
        for worker in self.gpu_workers:
            try:
                if hasattr(worker, 'process') and worker.process.is_alive():
                    Log.w(f"🛑 Sending termination signal to worker {worker.partition_id}")
                    worker.process.terminate()
            except Exception as e:
                Log.e(f"Error terminating worker: {e}")
                
    def _terminate_failed_workers(self, failed_gpu_id: int):
        """실패한 GPU의 worker들 즉시 종료"""
        Log.w(f"💀 Terminating workers on failed GPU {failed_gpu_id}")
        failed_workers = self._find_workers_on_gpu(failed_gpu_id)
        
        for worker in failed_workers:
            try:
                Log.w(f"🔥 Force killing worker {worker.partition_id} on GPU {failed_gpu_id}")
                if hasattr(worker, 'process'):
                    worker.process.kill()  # terminate 대신 kill로 강제 종료
                    worker.process.join(timeout=2)
            except Exception as e:
                Log.e(f"Error killing worker {worker.partition_id}: {e}")
                
    def _save_emergency_checkpoint(self):
        """긴급 체크포인트 저장"""
        try:
            Log.w("💾 Saving emergency checkpoint...")
            artifact_dir = self.config.get('__artifact_dir', '.')
            os.makedirs(artifact_dir, exist_ok=True)
            checkpoint_path = f"{artifact_dir}/emergency_checkpoint.pth"
            if self.module_online is None:
                Log.w("⚠️ No online model found; skipping checkpoint save")
                return

            checkpoint_payload = self._build_resume_compatible_checkpoint_payload(
                checkpoint_kind="emergency"
            )
            torch.save(checkpoint_payload, checkpoint_path)
            Log.i(f"✅ Emergency checkpoint saved to {checkpoint_path}")
        except Exception as e:
            Log.e(f"❌ Failed to save emergency checkpoint: {e}")

    def _save_healthy_checkpoint(self):
        """정상 구간의 롤백용 체크포인트 저장"""
        try:
            artifact_dir = self.config.get('__artifact_dir', '.')
            os.makedirs(artifact_dir, exist_ok=True)
            checkpoint_path = f"{artifact_dir}/healthy_checkpoint_latest.pth"

            if self.module_online is not None:
                save_start = time()
                checkpoint_payload = self._build_resume_compatible_checkpoint_payload(
                    checkpoint_kind="healthy"
                )
                checkpoint_payload['save_timestamp'] = save_start
                torch.save(checkpoint_payload, checkpoint_path)
                self.last_healthy_checkpoint_step = int(self.batch_count)
                save_duration_sec = time() - save_start
                file_size_bytes = os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else None
                self._save_event_seq += 1
                save_event = {
                    'event_type': 'checkpoint_save',
                    'timestamp_sec': time(),
                    'save_event_id': self._save_event_seq,
                    'batch_count': self.batch_count,
                    'checkpoint_interval': self.healthy_checkpoint_interval,
                    'checkpoint_path': checkpoint_path,
                    'save_duration_sec': save_duration_sec,
                    'file_size_bytes': file_size_bytes,
                    'pre_step_time_ms': self.last_completed_step_time_ms,
                }
                self._append_jsonl(self.save_events_file, save_event)
                self._pending_save_spike = {
                    'save_event_id': self._save_event_seq,
                    'batch_count': self.batch_count,
                    'pre_step_time_ms': self.last_completed_step_time_ms,
                }
                Log.i(f"✅ Healthy checkpoint saved: {checkpoint_path}")
        except Exception as e:
            Log.e(f"❌ Failed to save healthy checkpoint: {e}")

    def _latest_alpha_beta_snapshot_path(self) -> str:
        artifact_dir = self.config.get('__artifact_dir', '.')
        os.makedirs(artifact_dir, exist_ok=True)
        return f"{artifact_dir}/alpha_beta_latest.json"

    def _load_latest_alpha_beta_snapshot(self, gpu_ids: Optional[List[int]] = None) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]]]:
        snapshot_path = self._latest_alpha_beta_snapshot_path()
        if not os.path.exists(snapshot_path):
            return None, None

        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
        except Exception as e:
            Log.w(f"⚠️ Failed to read alpha/beta snapshot {snapshot_path}: {e}")
            return None, None

        alpha_payload = snapshot.get('alpha_comp')
        beta_payload = snapshot.get('beta_comm')
        if not isinstance(alpha_payload, dict) or not isinstance(beta_payload, dict):
            return None, None

        alpha = {int(k): float(v) for k, v in alpha_payload.items()}
        beta = {int(k): float(v) for k, v in beta_payload.items()}

        if gpu_ids is not None:
            filtered_gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
            alpha = {gpu_id: float(alpha.get(gpu_id, 1.0)) for gpu_id in filtered_gpu_ids}
            beta = {gpu_id: float(beta.get(gpu_id, 1.0)) for gpu_id in filtered_gpu_ids}

        return alpha, beta

    def _resolve_restart_alpha_beta(self, gpu_ids: List[int]) -> Tuple[Dict[int, float], Dict[int, float], str]:
        filtered_gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]

        snapshot_alpha, snapshot_beta = self._load_latest_alpha_beta_snapshot(filtered_gpu_ids)
        if snapshot_alpha is not None and snapshot_beta is not None:
            return snapshot_alpha, snapshot_beta, "runtime_snapshot"

        if hasattr(self, 'alpha_g') and hasattr(self, 'beta_g'):
            alpha = {gpu_id: float(self.alpha_g.get(gpu_id, 1.0)) for gpu_id in filtered_gpu_ids}
            beta = {gpu_id: float(self.beta_g.get(gpu_id, 1.0)) for gpu_id in filtered_gpu_ids}
            return alpha, beta, "tspipe_state"

        alpha = {gpu_id: 1.0 for gpu_id in filtered_gpu_ids}
        beta = {gpu_id: 1.0 for gpu_id in filtered_gpu_ids}
        return alpha, beta, "defaults"
            
    def _prepare_k_minus_1_restart(self, failed_gpu_id: int):
        """K-1 GPU 재분할 정보 준비"""
        Log.w(f"📋 Preparing K-1 GPU restart configuration (excluding GPU {failed_gpu_id})")
        
        try:
            # 사용 가능한 GPU 목록 (실패한 GPU 제외)
            available_gpus = [i for i in range(torch.cuda.device_count()) if i != failed_gpu_id]
            if not available_gpus:
                Log.e("❌ No available GPU remains after failure")
                return
            
            Log.i(f"📊 Available GPUs for restart: {available_gpus}")
            Log.i(f"📊 Original partition config: {self.original_partition_config}")

            import json
            artifact_dir = self.config.get('__artifact_dir', '.')
            os.makedirs(artifact_dir, exist_ok=True)
            restart_config_path = f"{artifact_dir}/emergency_restart_config.json"
            restart_alpha, restart_beta, coeff_source = self._resolve_restart_alpha_beta(available_gpus)
            resume_step = int(self.last_healthy_checkpoint_step) if int(self.last_healthy_checkpoint_step) > 0 else int(self.batch_count)

            # Persist minimal emergency payload first so external supervisor always has
            # a consumable restart config even if DP repartition crashes mid-path.
            restart_config = {
                'restart_type': 'hard_failure',
                'failed_gpu': failed_gpu_id,
                'available_gpus': available_gpus,
                'original_config': self.original_partition_config,
                'checkpoint_path': f"{artifact_dir}/healthy_checkpoint_latest.pth",
                'emergency_checkpoint_path': f"{artifact_dir}/emergency_checkpoint.pth",
                'failover_timestamp': time(),
                'step_id': resume_step,
                'alpha_comp': restart_alpha,
                'beta_comm': restart_beta,
                'alpha_beta_source': coeff_source,
            }
            with open(restart_config_path, 'w') as f:
                json.dump(restart_config, f, indent=2)
            Log.i(f"✅ Emergency restart base configuration saved to {restart_config_path}")
            Log.i(f"📊 Hard failover alpha/beta source: {coeff_source}")

            # Hard failure path: compute fresh K-1 contiguous partition via DP if planner is available.
            degraded_partition = None
            try:
                from planner.stage_time_predictor import StageTimePredictor

                predictor = StageTimePredictor()
                degraded_partition = predictor.solve_optimal_partition(
                    gpu_ids=available_gpus,
                    alpha_g=restart_alpha,
                    beta_g=restart_beta,
                )
                if degraded_partition is not None:
                    Log.i(
                        "🧮 Hard failover DP repartition complete: "
                        f"snet={degraded_partition.snet_partition}, "
                        f"tnet={degraded_partition.tnet_partition}, "
                        f"gpus={degraded_partition.gpu_assignment}"
                    )
            except Exception as e:
                Log.w(f"⚠️ Hard failover DP repartition unavailable, fallback to legacy metadata only: {e}")

            if degraded_partition is not None:
                restart_config['partition'] = {
                    'gpu_assignment': [int(g) for g in degraded_partition.gpu_assignment],
                    'snet_partition': [int(v) for v in degraded_partition.snet_partition],
                    'tnet_partition': [int(v) for v in degraded_partition.tnet_partition],
                }
                # Keep the same learned alpha/beta as soft failover, only excluding the failed GPU.
                filtered_alpha = {
                    int(gpu): float(restart_alpha.get(int(gpu), 1.0))
                    for gpu in degraded_partition.gpu_assignment
                }
                filtered_beta = {
                    int(gpu): float(restart_beta.get(int(gpu), 1.0))
                    for gpu in degraded_partition.gpu_assignment
                }
                restart_config['alpha_comp'] = filtered_alpha
                restart_config['beta_comm'] = filtered_beta
                restart_config['alpha_beta_source'] = coeff_source
                Log.i(
                    f"✅ Saved filtered alpha/beta from {coeff_source} "
                    f"for K-1 partition: {list(degraded_partition.gpu_assignment)}"
                )

            with open(restart_config_path, 'w') as f:
                json.dump(restart_config, f, indent=2)
                
            Log.i(f"✅ Emergency restart configuration saved to {restart_config_path}")
            
        except Exception as e:
            Log.e(f"❌ Failed to prepare restart configuration: {e}")

        # Failover 처리 시작 (기존 로직은 백업용으로 유지)
        # self._handle_gpu_failure(failure_event.gpu_id)

    def _on_process_failure(self, partition_id: int, failure_reason: str):
        """프로세스 실패 이벤트 핸들러"""
        if self._emergency_shutdown_started:
            Log.w(
                f"⚠️ Process failure ignored during active emergency shutdown "
                f"(partition={partition_id}, reason={failure_reason})"
            )
            return

        Log.e(f"💀 Process for partition {partition_id} failed: {failure_reason}")

        failed_gpu_id = None
        for worker in self.gpu_workers:
            if worker.partition_id == partition_id:
                failed_gpu_id = worker.device_id
                break
        if failed_gpu_id is None:
            failed_gpu_id = partition_id % max(1, self.num_devices)
        
        if self.experiment_logger:
            self.experiment_logger.log_failover_event(
                'gpu_failure',
                gpu_id=failed_gpu_id,
                partition_id=partition_id,
                details={
                    'failure_reason': failure_reason,
                    'source': 'process_health_monitor'
                }
            )

        Log.e(f"🛑 Escalating process failure to emergency shutdown (gpu={failed_gpu_id})")
        self._emergency_shutdown_and_failover(failed_gpu_id)

    def _handle_gpu_failure(self, failed_gpu_id: int):
        """GPU 실패 처리 및 복구"""
        if self.failover_in_progress:
            Log.w("Failover already in progress, skipping...")
            return
            
        self.failover_in_progress = True
        recovery_start_time = time()
        
        try:
            Log.i(f"🔄 Starting failover process for GPU {failed_gpu_id}")
            
            if self.experiment_logger:
                self.experiment_logger.log_failover_event('recovery_start', gpu_id=failed_gpu_id)
            
            # 1. 실패한 worker 식별 및 중지
            failed_workers = self._find_workers_on_gpu(failed_gpu_id)
            for worker in failed_workers:
                self._terminate_worker(worker)
                self.failed_workers[worker.partition_id] = worker
            
            # 2. K-1 방식: 남은 GPU들로 시스템 재구성 (백업 GPU 미사용)
            Log.i("⬇️ Executing K-1 reconfiguration (Soft Failure degrade approach)")
            Log.i(f"🔀 Reconfiguring system with remaining GPUs (excluding GPU {failed_gpu_id})")
            self._reconfigure_with_remaining_gpus()
            
            recovery_time_ms = (time() - recovery_start_time) * 1000
            
            Log.i(f"✅ Failover recovery completed in {recovery_time_ms:.2f}ms")
            
            if self.experiment_logger:
                self.experiment_logger.log_failover_event(
                    'recovery_complete', 
                    gpu_id=failed_gpu_id,
                    recovery_time_ms=recovery_time_ms,
                    new_config=self.current_partition_config
                )
                
        except Exception as e:
            Log.e(f"❌ Failover recovery failed: {e}")
            if self.experiment_logger:
                self.experiment_logger.log_message("❌ Failover 복구 실패", {'error': str(e)})
        finally:
            self.failover_in_progress = False

    def _find_workers_on_gpu(self, gpu_id: int) -> List[GpuWorker]:
        """특정 GPU에서 실행 중인 worker들 찾기"""
        failed_workers = []
        for worker in self.gpu_workers:
            if hasattr(worker, 'device_id') and worker.device_id == gpu_id:
                failed_workers.append(worker)
        return failed_workers

    def _terminate_worker(self, worker: GpuWorker):
        """Worker 프로세스 안전하게 종료"""
        try:
            Log.i(f"🛑 Terminating worker for partition {worker.partition_id}")
            
            # 현재 작업 완료 대기 (타임아웃 포함)
            self._wait_for_worker_completion(worker, timeout_seconds=10)
            
            # 프로세스 종료
            if hasattr(worker, 'process') and worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=5)
                
                if worker.process.is_alive():
                    worker.process.kill()
                    worker.process.join()
                    
            Log.i(f"✅ Worker for partition {worker.partition_id} terminated")
            
        except Exception as e:
            Log.e(f"Error terminating worker {worker.partition_id}: {e}")

    def _wait_for_worker_completion(self, worker: GpuWorker, timeout_seconds: int = 10):
        """Worker의 현재 작업 완료 대기"""
        # TODO: 실제 작업 완료 시그널 구현
        Log.d(f"Waiting for worker {worker.partition_id} completion...")
        sleep(2)  # 임시 대기

    def _restart_worker_on_gpu(self, partition_id: int, new_gpu_id: int):
        """새로운 GPU에서 worker 재시작"""
        Log.i(f"🚀 Restarting worker for partition {partition_id} on GPU {new_gpu_id}")
        
        # 새로운 worker 생성
        new_worker = self.spawn_new_gpu_workers(partition_id)  # TODO: GPU ID 지정 가능하도록 수정 필요
        
        # 기존 worker 교체
        for i, worker in enumerate(self.gpu_workers):
            if worker.partition_id == partition_id:
                self.gpu_workers[i] = new_worker
                break
        else:
            self.gpu_workers.append(new_worker)
            
        Log.i(f"✅ Worker restarted for partition {partition_id}")
        
        if self.experiment_logger:
            self.experiment_logger.log_failover_event(
                'worker_restart',
                partition_id=partition_id,
                details={'new_gpu_id': new_gpu_id}
            )

    def _reconfigure_with_remaining_gpus(self):
        """남은 GPU들로 시스템 재구성"""
        # 살아있는 GPU 목록 확인
        healthy_gpus = []
        if self.gpu_health_monitor:
            healthy_gpus = self.gpu_health_monitor.get_healthy_gpus()
        else:
            healthy_gpus = list(range(torch.cuda.device_count()))
        
        Log.i(f"🔀 Reconfiguring with {len(healthy_gpus)} healthy GPUs: {healthy_gpus}")
        
        # 새로운 파티션 설정 계산 (실제 GPU 목록 전달)
        new_config = self._calculate_new_partition_config(healthy_gpus)
        
        if new_config:
            self.current_partition_config = new_config
            Log.i("✅ System reconfigured successfully")
            
            if self.experiment_logger:
                self.experiment_logger.log_failover_event(
                    'repartition',
                    old_config=self.original_partition_config,
                    new_config=new_config,
                    details={'healthy_gpus': healthy_gpus}
                )
                self.experiment_logger.update_config(new_config)
        else:
            Log.e("❌ Failed to calculate new partition configuration")

    def _calculate_new_partition_config(self, healthy_gpu_ids: List[int]) -> Optional[Dict]:
        """DP 솔버를 사용한 최적 파티션 설정 계산 (Soft Failure degrade 방식)"""
        if not healthy_gpu_ids or len(healthy_gpu_ids) <= 0:
            return None
        
        try:
            # DP 솔버 import
            import sys
            import os
            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            if proj_root not in sys.path:
                sys.path.insert(0, proj_root)
            
            from benchmarks.soft_target.planner.stage_time_predictor import StageTimePredictor
            
            # 건강한 GPU들: alpha_g = beta_g = 1.0 (reprofiling 가정)
            alpha_g = {int(gpu_id): 1.0 for gpu_id in healthy_gpu_ids}
            beta_g = {int(gpu_id): 1.0 for gpu_id in healthy_gpu_ids}
            
            # DP 솔버 인스턴스 생성
            stage_time_predictor = StageTimePredictor()
            
            # DP 기반 최적 파티션 계산
            partition_config = stage_time_predictor.solve_optimal_partition(
                gpu_ids=healthy_gpu_ids,
                alpha_g=alpha_g,
                beta_g=beta_g
            )
            
            if partition_config is None:
                Log.w("⚠️ DP solver failed to compute partition, falling back to proportional split")
                return self._calculate_proportional_partition_config(len(healthy_gpu_ids))
            
            # PartitionConfig를 Dict로 변환
            new_config = {
                'online': list(partition_config.snet_partition),
                'target': list(partition_config.tnet_partition)
            }
            
            Log.i(f"🧮 DP solver computed optimal partition: online={new_config['online']}, target={new_config['target']}")
            Log.i(f"   GPU assignment: {partition_config.gpu_assignment}")
            return new_config
            
        except ImportError as e:
            Log.w(f"⚠️ Failed to import DP solver ({e}), falling back to proportional split")
            return self._calculate_proportional_partition_config(len(healthy_gpu_ids))
        except Exception as e:
            Log.e(f"❌ DP solver error: {e}, falling back to proportional split")
            return self._calculate_proportional_partition_config(len(healthy_gpu_ids))
    
    def _calculate_proportional_partition_config(self, num_healthy_gpus: int) -> Optional[Dict]:
        """Fallback: 비례적 분할"""
        if num_healthy_gpus <= 0:
            return None
        
        original_online = self.original_partition_config['online']
        original_target = self.original_partition_config['target']
        
        total_online_layers = sum(original_online)
        total_target_layers = sum(original_target)
        
        # 비례적 분할
        online_per_gpu = total_online_layers // num_healthy_gpus
        target_per_gpu = total_target_layers // num_healthy_gpus
        
        new_online = [online_per_gpu] * num_healthy_gpus
        new_target = [target_per_gpu] * num_healthy_gpus
        
        # 나머지 레이어들을 마지막 GPU에 할당
        remaining_online = total_online_layers - sum(new_online)
        remaining_target = total_target_layers - sum(new_target)
        
        if remaining_online > 0:
            new_online[-1] += remaining_online
        if remaining_target > 0:
            new_target[-1] += remaining_target
        
        Log.i(f"📊 Proportional partition: online={new_online}, target={new_target}")
        
        return {
            'online': new_online,
            'target': new_target
        }

    def _restart_failed_worker(self, partition_id: int):
        """실패한 worker 재시작"""
        Log.i(f"🔄 Restarting failed worker for partition {partition_id}")
        
        # 기존 worker 찾기 및 제거
        failed_worker = None
        for i, worker in enumerate(self.gpu_workers):
            if worker.partition_id == partition_id:
                failed_worker = worker
                del self.gpu_workers[i]
                break
        
        if failed_worker:
            # 새로운 worker 생성
            new_worker = self.spawn_new_gpu_workers(partition_id)
            self.gpu_workers.append(new_worker)
            
            Log.i(f"✅ Worker restarted for partition {partition_id}")

    def _simulate_gpu_failure_if_scheduled(self):
        """예약된 GPU 실패 시뮬레이션 실행"""
        # 디버깅을 위한 상세 로깅
        if self.batch_count % 5 == 0:  # 5배치마다 상태 로깅
            Log.i(f"🔍 Failure simulation check: batch_count={self.batch_count}, target_fail_gpu={self.target_fail_gpu}, fail_after_batches={self.fail_after_batches}, failure_simulated={self.failure_simulated}, gpu_health_monitor={'exists' if self.gpu_health_monitor else 'None'}")
        
        if (self.target_fail_gpu >= 0 and 
            self.batch_count >= self.fail_after_batches and 
            not self.failure_simulated and
            self.gpu_health_monitor):
            
            Log.w(f"💣 Simulating GPU {self.target_fail_gpu} failure after {self.batch_count} batches")
            
            success = self.gpu_health_monitor.force_gpu_failure(
                self.target_fail_gpu, 
                failure_type="scheduled_simulation"
            )
            
            if success:
                self.failure_simulated = True
                Log.w(f"✅ GPU {self.target_fail_gpu} failure simulation marked successful")
                if self.experiment_logger:
                    self.experiment_logger.log_message("💣 GPU 실패 시뮬레이션 실행", {
                        'gpu_id': self.target_fail_gpu,
                        'batch_count': self.batch_count
                    })
            else:
                Log.e(f"❌ GPU {self.target_fail_gpu} failure simulation failed")
        elif self.batch_count >= self.fail_after_batches:
            # 조건이 안 맞는 이유를 로깅
            reasons = []
            if self.target_fail_gpu < 0:
                reasons.append(f"target_fail_gpu={self.target_fail_gpu}")
            if self.failure_simulated:
                reasons.append("already_simulated")
            if not self.gpu_health_monitor:
                reasons.append("no_health_monitor")
            if reasons:
                Log.w(f"🚫 Failure simulation skipped at batch {self.batch_count}: {', '.join(reasons)}")

    def cleanup_failover_system(self):
        """Failover 시스템 정리"""
        if not self.failover_enabled:
            return
            
        Log.i("🧹 Cleaning up failover system...")
        
        if self.gpu_health_monitor:
            self.gpu_health_monitor.stop_monitoring()
            
        if self.process_health_monitor:
            self.process_health_monitor.stop_monitoring()
            
        if self.experiment_logger:
            result_dir = finalize_experiment_logger()
            Log.i(f"📊 Experiment results saved to: {result_dir}")
            
        Log.i("✅ Failover system cleanup completed")

    def join_workers(self):
        Log.i("Waiting until workers finish their jobs...")
        
        # Failover 모니터링 시작 (primary node에서만)
        if self.rank == 0 and self.failover_enabled:
            self._start_failover_monitoring()
        
        for worker in self.gpu_workers:
            worker.process.join()
            
        # Failover 시스템 정리
        if self.rank == 0:
            self.cleanup_failover_system()

    @property
    def is_primary(self):
        return self.rank == 0

    def spawn_new_gpu_workers(self, partition_id: int):
        """Spawn a new GPU worker for partition `partition_id`."""
        return GpuWorker(partition_id,
                         self.num_ubatches,
                         self.num_bwd_ubatches,
                         CommunicatorParam(self.args.ip, self.total_world_size, partition_id,
                                           nccl_port=self._nccl_port, rpc_port=self._rpc_port), 
                         self.optimizer, 
                         self.momentum,
                         self.loss_fn,
                         self.target_update_fn)
 
    def _thread_scheduler(self):
        self.running = PipelineRunState.RUNNING
        self.processed_batch_idx = 1
        set_terminated_partition = set() # self.num_partitions와 같아지면 종료 (모든 partition이 종료되면)
        gpipe_emulation_enabled = self.config['gpipe_emulation']['enabled']
        num_skip_initial_staleness = self.config['optimizer']['num_skip_initial_staleness']
        epoch = 0

        for schedules in self.task_scheduler.schedule_generator(): # 한 사이클의 작업들 리스트가 training 끝날 때까지 무한생성
            # 🔥 긴급 중단 체크
            if self._shutdown_scheduler:
                Log.e("[SCHEDULER] 🛑 Emergency shutdown flag detected. Stopping scheduler immediately.")
                self.running = PipelineRunState.STOPPED
                return
                
            if len(set_terminated_partition) == self.num_partitions:
                print("[SCHEDULER] Termination signal is sent to all partitions. Stopping scheduler loop.")
                self.running = PipelineRunState.STOPPED
            
            if self.running == PipelineRunState.STOPPED:
                print("[SCHEDULER] Scheduler thread exiting.")
                return

            for sched in schedules: # 4개 리스트. 한 사이클 동안 각 파티션이 무슨 작업을 할지
                if sched is not None:
                    is_target = sched.is_target
                    terminate = False
                    if self.running != PipelineRunState.RUNNING and sched.batch_idx > self.last_batch_idx:
                        # print(f"[LOG] Scheduler terminating at batch {sched.batch_idx}, reason: running={self.running}")
                        terminate = True
                        
                    if terminate:
                        # print(f"[SCHEDULER] Terminating scheduler at batch {sched.batch_idx} (last_batch_idx={self.last_batch_idx})")
                        if sched.j not in set_terminated_partition:
                            # task_terminate = GpuTask(TaskType.TASK_TERMINATE, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target)
                            # print(f"[SCHEDULER] Issuing TASK_TERMINATE to partition {sched.j}")
                            # self.schedule_task(task_terminate)
                            set_terminated_partition.add(sched.j)
                        continue
                    
                    stream_compute = StreamDescriptor(sched.j, StreamType.STREAM_DEFAULT_COMPUTE)
                    stream_from = StreamDescriptor(sched.j, StreamType.STREAM_COPY_BATCH_FROM)

                    if sched.optimize:
                        if sched.i != 0:
                            continue
                        o = self.did_epoch_terminate(sched.batch_idx)
                        new_momentum, new_lr = None, None
                        last_optim_partition = self.num_partitions - 1
                        if gpipe_emulation_enabled or sched.batch_idx <= num_skip_initial_staleness:
                            last_optim_partition = 0
                        if sched.batch_idx in self.scheduled_momentum_update:
                            new_momentum = self.scheduled_momentum_update[sched.batch_idx]
                            if sched.j == last_optim_partition:
                                self.scheduled_momentum_update.pop(sched.batch_idx)
                        if sched.batch_idx in self.scheduled_lr_update:
                            new_lr = self.scheduled_lr_update[sched.batch_idx]
                            if sched.j == last_optim_partition:
                                self.scheduled_lr_update.pop(sched.batch_idx)
                        task_compute_optimize_gpu = GpuTask(TaskType.TASK_COMPUTE_OPTIMIZE_GPU, sched.batch_idx, sched.view_idx, \
                                                            sched.i, sched.j, is_target, optimizer_step=o, new_momentum=new_momentum, new_lr=new_lr)
                        self.schedule_task(task_compute_optimize_gpu)
                        if o:
                            epoch += 1
                        continue

                    if self.tspipe_mode == TSPipeMode.SUPERVISED_MOMENTUM:
                        view_bool = True if sched.view_idx == 1 else False
                        if view_bool != sched.is_target:
                            # print(f"Skipping transmission of {sched}")
                            if sched.model_update_ver is not None and sched.model_update_ver > 0:
                                task_model_copy = GpuTask(TaskType.TASK_COPY_MODEL, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, 
                                dst_stream=stream_compute, src_stream=stream_copy_dst)
                                self.schedule_task(task_model_copy)
                            continue
                        else:
                            pass
                            # print(f"Allowing transmission of {sched}")

                    if sched.backprop:
                        is_target = False
                        task_backward = GpuTask(TaskType.TASK_COMPUTE_BACKWARD, sched.batch_idx, sched.view_idx, \
                                                sched.i, sched.j, is_target, src_stream=stream_compute, wait_stream=stream_from, asymmetric=(self.tspipe_mode==TSPipeMode.SUPERVISED_MOMENTUM))
                        num_bwd_batch = self.num_bwd_ubatches if not gpipe_emulation_enabled and sched.batch_idx > num_skip_initial_staleness else self.num_ubatches
                        if self.tspipe_mode == TSPipeMode.SELF_SUPERVISED_MOMENTUM and sched.j == self.num_partitions - 1 and sched.i == num_bwd_batch - 1 and sched.view_idx == 1 or \
                           self.tspipe_mode == TSPipeMode.SUPERVISED_MOMENTUM      and sched.j == self.num_partitions - 1 and sched.i == num_bwd_batch - 1:
                            task_loss = GpuTask(TaskType.TASK_COMPUTE_LOSS, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, src_stream=stream_compute, 
                                                asymmetric=self.tspipe_mode==TSPipeMode.SUPERVISED_MOMENTUM, epoch=epoch, scatter_gather_fn=self.scatter_gather_fn)
                            self.schedule_task(task_loss)
                        self.schedule_task(task_backward)

                        if sched.j > 0:
                            task_copy_grad_out = GpuTask(TaskType.TASK_COPY_GRAD_OUT, sched.batch_idx, sched.view_idx, \
                                                        sched.i, sched.j, is_target, src_stream=stream_compute, wait_stream=stream_from, asymmetric=(self.tspipe_mode==TSPipeMode.SUPERVISED_MOMENTUM))
                            self.schedule_task(task_copy_grad_out)

                            stream_compute = StreamDescriptor(sched.j-1, StreamType.STREAM_DEFAULT_COMPUTE)
                            stream_copy_dst = StreamDescriptor(sched.j-1, StreamType.STREAM_COPY_BATCH_TO)
                            stream_copy_src = StreamDescriptor(sched.j, StreamType.STREAM_COPY_BATCH_FROM)
                            task_grad_copy = GpuTask(TaskType.TASK_COPY_GRAD, sched.batch_idx, sched.view_idx, sched.i, sched.j-1, is_target, 
                                                     dst_stream=stream_copy_dst, src_stream=stream_copy_src, wait_stream=stream_compute)
                            self.schedule_task(task_grad_copy)
                    else:
                        stream_copy_dst = StreamDescriptor(sched.j, StreamType.STREAM_COPY_BATCH_TO)
                        stream_copy_src = StreamDescriptor(sched.prev_partition, StreamType.STREAM_COPY_BATCH_FROM)
                        stream_copy_to_next = StreamDescriptor(sched.j, StreamType.STREAM_COPY_BATCH_FROM) 
                        task_compute = GpuTask(TaskType.TASK_COMPUTE_FORWARD, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, 
                                               dst_stream=stream_copy_dst, src_stream=stream_compute, wait_stream=stream_copy_to_next, scatter_gather_fn=self.scatter_gather_fn)
                        task_batch_copy_out = GpuTask(TaskType.TASK_COPY_BATCH_OUT, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, asymmetric=(self.tspipe_mode==TSPipeMode.SUPERVISED_MOMENTUM))
                        if sched.model_update_ver is not None and sched.model_update_ver > 0:
                            task_model_copy = GpuTask(TaskType.TASK_COPY_MODEL, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, 
                            dst_stream=stream_compute, src_stream=stream_copy_dst)
                            self.schedule_task(task_model_copy)

                        task_batch_copy = GpuTask(TaskType.TASK_COPY_BATCH, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, 
                                                  dst_stream=stream_copy_dst, src_stream=stream_copy_src, wait_stream=stream_compute)
                        self.schedule_task(task_batch_copy)
                        self.schedule_task(task_compute)
                        if sched.j < self.num_partitions - 1:
                            self.schedule_task(task_batch_copy_out)

    def did_epoch_terminate(self, batch_id: int) -> bool:
        return batch_id in self.batch_q.epoch_boundaries

    def schedule_task(self, gpu_task: GpuTask):
        self.highest_scheduled_batch_id = max(self.highest_scheduled_batch_id, gpu_task.batch_id)
        
        if gpu_task.task_type == TaskType.TASK_FEED_BATCH:
            self.batch_count += 1
            if self.periodic_checkpoint_enabled and self.healthy_checkpoint_interval > 0 and self.batch_count % self.healthy_checkpoint_interval == 0:
                self._save_healthy_checkpoint()
            if self.failover_enabled:
                try:
                    self._simulate_gpu_failure_if_scheduled()
                except Exception as e:
                    Log.e(f"Error in failure simulation check: {e}")
        
        if gpu_task.task_type == TaskType.TASK_TERMINATE:
            if gpu_task.partition_id is not None and gpu_task.partition_id >= 0:
                future = self.comm.send(f'task_{gpu_task.partition_id}', gpu_task)
                self.terminate_futures.append(future)
            else:
                self.local_worker.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_BATCH or gpu_task.task_type == TaskType.TASK_COPY_BATCH_OUT:
            self.comm.send(f'task_{gpu_task.partition_id}', gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_GRAD or gpu_task.task_type == TaskType.TASK_COPY_MODEL:
            self.comm.send(f'task_{gpu_task.partition_id}', gpu_task)
        elif gpu_task.task_type == TaskType.TASK_FEED_BATCH:
            self.local_worker.schedule(gpu_task)
        else:
            self.comm.send(f'task_{gpu_task.partition_id}', gpu_task)

    def start_pipeline(self) -> None:
        """Starts pipeline that schedules computing tasks to each GPU."""
        self.thread = Thread(
            target=self._thread_scheduler,
            args=[],
            daemon=True,
        )
        self.thread.name='SchedulerThread'
        self.thread.start()

    def feed(self, view_1: TensorOrTensors, view_2: TensorOrTensors, view_target: Optional[TensorOrTensors] = None) -> Optional[torch.Tensor]:
        """Feed new dataset to pipeline."""
        assert self.running != PipelineRunState.PENDING_STOP
        assert self.rank == 0, "Must be primary to feed batch"

        if (self.max_step_profiling is not None) and (self.batch_q.next_id >= self.max_step_profiling):
            return None


        # if (self.max_step_profiling is not None) and (self.batch_q.next_id >= self.max_step_profiling):
            # print(f"[feed] Max step {self.max_step_profiling} reached. Skipping further feed.")
            # return None

        # sanity check
        if torch.is_tensor(view_1):
            batch_ops.check(view_1)
        if torch.is_tensor(view_2):
            batch_ops.check(view_2)
        
        # split into multiple batches for gradient accumulation
        # only apply gradient accumulation if microbatch size is geq than 4
        assert self.scatter_gather_fn.batch_size(view_1) == self.scatter_gather_fn.batch_size(view_2)
        gradient_accumulation = self.config['optimizer']['gradient_accumulation']
        if self.scatter_gather_fn.batch_size(view_1) >= 4 * self.num_ubatches * gradient_accumulation:
            minibatches_view_m1 = self.scatter_gather_fn.scatter(view_1, gradient_accumulation)
            minibatches_view_m2 = self.scatter_gather_fn.scatter(view_2, gradient_accumulation)
        else:
            minibatches_view_m1 = self.scatter_gather_fn.scatter(view_1, 1)
            minibatches_view_m2 = self.scatter_gather_fn.scatter(view_2, 1)
        
        # start the pipeline in another thread if not running
        if self.running == PipelineRunState.STOPPED:
            self.start_pipeline()

        torch.cuda.profiler.start()
        
        # inject batches to batch_q and wait for the loss
        for view_m1, view_m2 in zip(minibatches_view_m1, minibatches_view_m2):
            if (self.max_step_profiling is not None) and (self.batch_q.next_id >= self.max_step_profiling):
                # Log.i(f"[feed] Max step {self.max_step_profiling} reached. Skipping further feed.")
                return None

            batches_1 = self.scatter_gather_fn.scatter(view_m1.tensor_or_tensors, self.num_ubatches)
            batches_2 = self.scatter_gather_fn.scatter(view_m2.tensor_or_tensors, self.num_ubatches)

            step_start = time()

            bid = self.batch_q.next_id
            with Operation('feed_api', batch_idx=bid):
                if self.tspipe_mode == TSPipeMode.SELF_SUPERVISED_MOMENTUM or \
                   (hasattr(self.tspipe_mode, 'value') and self.tspipe_mode.value == 0) or \
                   (hasattr(self.tspipe_mode, 'name') and self.tspipe_mode.name == 'SELF_SUPERVISED_MOMENTUM'):
                    batches_lists = [batches_1, batches_2, batches_2, batches_1] # no clone needed here
                    asymmetric = False
                    assert view_target is None, "Self-supervised learning does not support labels."
                    batch_view_target = None
                elif self.tspipe_mode == TSPipeMode.SUPERVISED_MOMENTUM or \
                     (hasattr(self.tspipe_mode, 'value') and self.tspipe_mode.value == 1) or \
                     (hasattr(self.tspipe_mode, 'name') and self.tspipe_mode.name == 'SUPERVISED_MOMENTUM'):
                    batches_lists = [batches_1, batches_2] # no clone needed here
                    asymmetric = True
                    assert view_target is not None, "Supervised learning requires labels."
                    batch_view_target = Batch(view_target)
                else:
                    assert False, f"Invalid Mode: {self.tspipe_mode} (type: {type(self.tspipe_mode)})"
                batch_idx = self.batch_q.get_new_batch_id()
                self.last_batch_idx = batch_idx
                self.schedule_task(GpuTask(TaskType.TASK_FEED_BATCH, batch_idx, 0, 0, 0, 0, batch_list=batches_lists, asymmetric=asymmetric, label_batch=batch_view_target))
            loss = self.wait_forward()
            step_time_ms = (time() - step_start) * 1000.0
            if loss is not None:
                completed_step_id = self.forward_complete_batch_idx
                self.last_completed_step_time_ms = step_time_ms
                if self.checkpoint_benchmark_enabled:
                    self._record_step_metric(completed_step_id, step_time_ms)

        # loss may be None for first few initial iterations
        return loss

    def feed_epoch(self):
        print(f"epoch will end at batch {self.last_batch_idx+1}")
        self.batch_q.report_epoch_boundary()

    def stop(self) -> None:
        assert self.rank == 0

        # =================== Scheduler thread exit ===================
        # print("[STOP] stop() called by rank 0")
        self.running = PipelineRunState.PENDING_STOP
        # print(f"[STOP] PipelineRunState set to PENDING_STOP")
        
        # print("[STOP] Calling batch_q.stop()")
        self.batch_q.stop()
        # print("[STOP] batch_q.stop() returned")
        
        
        # FIXME: Alredy done when (running = PENDING_STOP --> terminate = True) in _thread_scheduler()
        for p_id in range(self.num_partitions):
            task_terminate = GpuTask(TaskType.TASK_TERMINATE, 0, 0, 0, p_id, False)
            # print(f"[DEBUG] scheduling TERMINATE to partition {p_id}")
            self.schedule_task(task_terminate)

        task_terminate = GpuTask(TaskType.TASK_TERMINATE, 0, 0, 0, -1, False)
        # print(f"[DEBUG] scheduling TERMINATE to partition {-1}")
        self.schedule_task(task_terminate)
        
        
        # for fut in self.terminate_futures:
        #     print(f"[DEBUG] FUTURE {fut}, done={fut.done()}")
        #     fut.wait()  # 또는 .result() 로 exception까지 체크

        self.thread.join()
        # if self.thread.is_alive():
        #     print("[DEBUG] SchedulerThread thread did NOT join gracefully")
        # else:
        #     print("[DEBUG] SchedulerThread thread joined successfully")

        torch.cuda.profiler.stop()
        
        # for idx, channel in enumerate(self.model_out_channels):
        #     print(channel.name)
        #     new_state_dict = channel.recv()
        #     print(f"got new_state_dict f{get_shape(new_state_dict)}")
        #     self.partitions_target[idx].load_state_dict(new_state_dict)

        self.local_worker.stop() # Stop and join local worker thread
        sleep(10)

        # print("[DEBUG] BEFORE profiler_delegate_worker.join()")
        # self.profiler_delegate_worker.join()
        # print("[DEBUG] AFTER profiler_delegate_worker.join()")

        # print("Terminating GPU workers")
        for w in self.gpu_workers:
            w.join()
        
        
        # print("Terminating comm")
        self.comm.finish(False)
        sleep(10)

    def wait_forward(self) -> Optional[float]:
        """Blocks the current thread until the next computation for loss is complete,
        provided batch feeds are sufficient enough to fully utilize the pipeline.
        Immediately returns None if the batch feeds are insufficient. (i.e., in the beginning of training)

        Returns:
            Computed loss, or None
        """
        try:
            forward_batch_id, loss = self.forward_out_queue.get_nowait()
            self.forward_complete_batch_idx = forward_batch_id
            return loss
        except Empty:
            pass

        # if batch feeds are insufficient, return immediately
        if self.batch_q.next_id < self.forward_complete_batch_idx + 8:
            Log.d(f"Wait_forward IMM2: Next batch to arrive {self.batch_q.next_id}, Last enqueued batch {self.processed_batch_idx}, Last forward complete batch {self.forward_complete_batch_idx}")
            return None

        # wait for the loss computation result
        forward_batch_id, loss = self.forward_out_queue.get()
        self.forward_complete_batch_idx = forward_batch_id
        return loss

    def update_momentum(self, new_momentum: float):
        assert self.batch_q.next_id > self.highest_scheduled_batch_id, f"Batch {self.highest_scheduled_batch_id} has been already scheduled. Trying to schedule {self.batch_q.next_id}"
        self.scheduled_momentum_update[self.batch_q.next_id] = new_momentum
        Log.i(f"Scheduling momentum update at batch {self.batch_q.next_id} to {new_momentum}")

    def update_lr(self, new_lr: Union[float, List[float]]):
        assert self.batch_q.next_id > self.highest_scheduled_batch_id, f"Batch {self.highest_scheduled_batch_id} has been already scheduled. Trying to schedule {self.batch_q.next_id}"
        self.scheduled_lr_update[self.batch_q.next_id] = new_lr
        Log.i(f"Scheduling lr update at batch {self.batch_q.next_id} to {new_lr}")

    def split_module(self, balance_online: Iterable[int], balance_target: Iterable[int]) -> Tuple[List[torch.nn.Sequential], List[torch.nn.Sequential]]:
        """Splits `self.module_online`+`self.module_predictor` and `self.module_target`,
         and stores them into `self.partitions_online` and `self.partitions_target`.

        Returns: 
            A tuple of (`List[model]` for online module, `List[model]` for target module)
            
        Raises:
            BalanceError:
                wrong balance
        """
        module_online, module_target, module_predictor = self.module_online, self.module_target, self.module_predictor
        balance_online = list(balance_online)
        balance_target = list(balance_target)

        if len(module_online) + (len(module_predictor) if module_predictor is not None else 0) != sum(balance_online):
            raise BalanceError('online module and sum of balance have different length '
                            f'(module: {len(module_online)}, sum of balance: {sum(balance_online)})')

        if len(module_target) != sum(balance_target):
            raise BalanceError('target module and sum of balance have different length '
                            f'(module: {len(module_target)}, sum of balance: {sum(balance_target)})')

        if any(x <= 0 for x in balance_online):
            raise BalanceError(f'all balance numbers must be positive integer (balance: {balance_online})')
        if any(x <= 0 for x in balance_target):
            raise BalanceError(f'all balance numbers must be positive integer (balance: {balance_target})')


        def split(module_children: Iterable[torch.nn.Module], balance: List[int]) -> List[torch.nn.Sequential]:
            j = 0
            partitions = []
            layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
            for name, layer in module_children:
                layer_input: torch.nn.Module = layer

                if len(layers) == 0:
                    # make this layer as leaf
                    for param in layer_input.parameters():
                        param.detach_()
                        param.requires_grad = True
                        assert param.is_leaf
                    
                layers[name] = layer_input

                if len(layers) == balance[j]:
                    # Group buffered layers as a partition.
                    partition = torch.nn.Sequential(layers)
                    partitions.append(partition)

                    # Prepare for the next partition.
                    layers.clear()
                    j += 1
            print([len(part) for part in partitions])
            return cast(List[torch.nn.Sequential], partitions)

        if module_predictor is not None:
            partition_online = split(chain(module_online.named_children(), module_predictor.named_children()), balance_online)
        else:
            partition_online = split(module_online.named_children(), balance_online)
        partition_target = split(module_target.named_children(), balance_target)

        return partition_online, partition_target
