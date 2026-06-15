"""GPU Worker"""
from collections import deque, defaultdict
from copy import deepcopy
from queue import Empty
from queue import Queue as LocalQueue
from threading import Condition, Thread
from time import monotonic, sleep, time
from typing import Callable, Dict, List, Optional, Union
from multiprocessing import Process, Queue
from .profiler_utils import init_gpu_task_profiler, stop_gpu_task_profiler
import torch
from torch.futures import Future
import threading
import traceback
import os


import tspipe.multiprocessing as mp
from tspipe.affinity_manager import AffinityManager
from tspipe.batch_ops import Microbatch
from tspipe.communicator import (Communicator, CommunicatorParam,
                                 DistributedQueue)
from tspipe.gpu_context import (ActivationStorage, DeviceState, GpuTaskContext,
                                LocalTaskContext, ParamStorage,
                                StreamDescriptor, StreamType)
from tspipe.gpu_task import GpuTask, TaskType
from tspipe.logger import Log
from tspipe.profiler import remote_profile_init
from tspipe.utils import new_stream, traverse_tensor_map


class SubGpuWorker():
    def __init__(self, base_worker: 'GpuWorker', role: str = ''):
        self.base_worker = base_worker
        self.device = base_worker.device
        self.thread: Optional[Thread] = None
        self.task_queue: mp.Queue[Optional[GpuTask]] = mp.Queue()
        self.current_task = None
        self.last_task = None
        self.last2_task = None
        self.running = True
        self.debug_info = None
        self.backward_last_lst = [0]
        self.role = role
        self.cv: Optional[Condition] = None
        self.backward_cv: Optional[Condition] = None
        self.complete_task_queue: Optional[Union[LocalQueue, mp.Queue]] = None
        self.complete_task_set: List[GpuTask] = []

    def post_init(self):
        self.cv = Condition()
        self.backward_cv = Condition()
        if self.complete_task_queue is None:
            self.complete_task_queue = LocalQueue()
        self.stashed_task_deque: deque[GpuTask] = deque()

    def start(self):
        self.post_init()
        role_name = f'GPU{self.device.index}{self.role[0:1].capitalize() + self.role[1:]}Thread'
        Log.d(f'{role_name} is starting...')
        self.thread = Thread(target=self._thread, args=())
        self.thread.name = role_name
        self.thread.start()
        Log.d(f'Thread {role_name} is running')
        return self.thread

    def wait_for_task_complete(self, task: 'GpuTask'):
        while True:
            if task in self.complete_task_set:
                self.complete_task_set.remove(task)
                return
            t = self.complete_task_queue.get()
            self.complete_task_set.append(t)

    def schedule(self, task: 'GpuTask'):
        task.schedule(self)
        self.task_queue.put(task)
        if self.base_worker.worker_level_cv:
            with self.base_worker.worker_level_cv:
                self.base_worker.worker_level_cv.notify_all()

        if self.cv is not None:
            self.cv.acquire()
            self.cv.notify()
            self.cv.release()

    def stop(self):
        if self.cv is None:
            self.cv = Condition()
        # self.cv.acquire()
        # self.running = False
        # self.cv.notify()
        # self.cv.release()
        # self.task_queue.put(None)

        self.running = False
        self.task_queue.put(None)
        with self.cv:
            self.cv.notify_all()
        if self.base_worker.worker_level_cv:
            with self.base_worker.worker_level_cv:
                self.base_worker.worker_level_cv.notify_all()

    def log(self, *args):
        Log.d(" ".join([f"{self.thread.name if self.thread is not None else 'Uninitialized'}:\t", *args]))

    def log_(self, *args):
        Log.d(" ".join([f"{self.thread.name if self.thread is not None else 'Uninitialized'}:\t", *args]))

    def log__(self, *args):
        Log.d(" ".join([f"{self.thread.name if self.thread is not None else 'Uninitialized'}:\t", *args]))

    def debug_stat(self):
        self.log_(f"Current: {self.current_task}")
        self.log_(f"Last   : {self.last_task}")

    def get_task_ctx(self, task):
        return self.base_worker.list_context[task.ubatch_id]

    def _thread(self):

        def debug_print_condition(task: GpuTask):
            '''For debugging purposes'''
            return False

        while self.running:
            # Block until receiving task
            try:
                task: Optional[GpuTask] = self.task_queue.get(timeout=0.1)
            except Empty:
                continue
            
            if not self.running and self.task_queue.empty():
                print(f"Terminating thread {self.thread.name}")
                return
            
            if task is None or task.task_type == TaskType.TASK_TERMINATE:
                print(f"[{self.role}] Received None task. Terminating thread.")
                self.running = False
                # break
                return

            while not task.check_precondition(self.get_task_ctx(task)):
                with self.base_worker.worker_level_cv:
                    self.base_worker.worker_level_cv.wait()
                    if not self.running:
                        print(f"Terminating thread {self.thread.name}")
                        return

            self.current_task = task
            assert task is not None
            if debug_print_condition(task):
                self.log__(f"Starting {task}")
            ctx: GpuTaskContext = self.base_worker.list_context[task.ubatch_id]

            task_start = time()
            task.run(ctx)
            task_elapsed_ms = (time() - task_start) * 1000.0

            if hasattr(self.base_worker, "update_task_baseline"):
                self.base_worker.update_task_baseline(task, task_elapsed_ms)
            if debug_print_condition(task):
                self.log__(f"Finished {task}")
            task.completed = True
            self.last2_task = self.last_task
            self.last_task = self.current_task
            self.current_task = None
            
            with self.base_worker.worker_level_cv:
                self.base_worker.worker_level_cv.notify_all()
                
            del task
        return


class SubCpuWorker(SubGpuWorker):
    def __init__(self, base_worker: 'LocalWorker', role: str = ''):
        super().__init__(base_worker, role)
        self.base_worker = base_worker

    def start(self):
        self.stashed_task_deque: deque[GpuTask] = deque()
        self.thread = Thread(target=self._thread, args=())
        self.thread.name = f'CPU0{self.role[0:1].capitalize() + self.role[1:]}Thread'
        self.thread.start()
        return self.thread


class BaseWorker():
    @staticmethod
    def gpu_device_id_to_internal_device_id(device_id: int) -> int:
        # For best NVLink Performance
        device_count = torch.cuda.device_count()
        assert device_id < device_count
        if device_count < 4:
            return device_id
        elif device_count == 4:
            return [0, 1, 2, 3][device_id]
        elif device_count == 8:
            return [0, 3, 2, 1, 5, 6, 7, 4][device_id]
        else:
            return device_id

    @staticmethod
    def partition_id_to_device_id(partition_id: int) -> int:
        import os
        # 사용 가능한 GPU 개수 파악 (CUDA_VISIBLE_DEVICES 고려)
        visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible_gpus:
            num_gpus = len([g.strip() for g in visible_gpus.split(',') if g.strip()])
        else:
            num_gpus = torch.cuda.device_count()
        return partition_id % num_gpus

    @staticmethod
    def partition_id_to_internal_device_id(partition_id: int) -> int:
        return BaseWorker.gpu_device_id_to_internal_device_id(BaseWorker.partition_id_to_device_id(partition_id))

    def __init__(self, partition_id: Optional[int], num_ubatch: Optional[int], num_bwd_ubatch: Optional[int],
                 communicator_param: Optional[CommunicatorParam], list_context: List[GpuTaskContext]):

        self.device: Optional[torch.device] = None
        self.device_id: Optional[int] = None
        self.internal_device_id: Optional[int] = None
        self.partition_id = partition_id

        if partition_id is not None and partition_id >= 0:
            # allocate gpu
            self.device_id = partition_id % torch.cuda.device_count()
            self.internal_device_id = BaseWorker.gpu_device_id_to_internal_device_id(self.device_id)
            self.device = torch.device('cuda', self.internal_device_id)
            Log.v(f"Partition ID = {self.partition_id}, Device ID = {self.device_id}, Device = {self.device}")
            torch.cuda.set_device(self.device)
        else:
            self.partition_id = -1
        self.num_ubatch = num_ubatch
        self.num_bwd_ubatch = num_bwd_ubatch
        self.process = None
        self.list_context = list_context

        self.communicator_param = communicator_param

        self.complete_task_set = []
        self.complete_task_set_cv = None
        self.running = True

        self.oob_resp_queue = mp.Queue()

        self.worker_level_cv: Optional[Condition] = None

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError

    def wait_until_ready(self):
        while True:
            resp = self.oob_resp_queue.get()
            if resp == 'ready':
                return

    def init_distributed_comm(self):
        self.comm = Communicator(self.device, self.communicator_param)
        Log.v(f"DistributedWorker {self.communicator_param.rank} is now ready.")

        num_partition = self.communicator_param.num_partition
        num_ubatch = max(self.num_ubatch, self.num_bwd_ubatch)
        partition_id = self.partition_id

        assert partition_id == self.communicator_param.rank

        self.num_partition = num_partition

        for cname in [f'init_config_{partition_id}', f'init_model_{partition_id}', f'task_{partition_id}']:
            self.comm.create_channel(cname, self.comm.scheduler_process_rank)

        if partition_id < num_partition - 1:
            for cname in [f'batch_{partition_id+1}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
                self.comm.create_channel(cname, partition_id+1)

        if partition_id > 0:
            for cname in [f'batch_{partition_id}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
                self.comm.create_channel(cname, partition_id-1)
        else:
            for cname in [f'batch_{partition_id}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
                self.comm.create_channel(cname, self.comm.world_size - 1)

        if partition_id > 0:
            for cname in [f'grad_{partition_id-1}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
                self.comm.create_channel(cname, partition_id-1)

        for cname in [f'grad_{partition_id}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
            self.comm.create_channel(cname, partition_id+1)

        if partition_id == num_partition - 1:
            cname = 'loss_out'
            self.comm.create_channel(cname, self.comm.scheduler_process_rank)

            cname = 'label_feed'
            self.comm.create_channel(cname, self.comm.scheduler_process_rank)

        cname = f'log_{partition_id}'
        self.comm.create_channel(cname, self.comm.scheduler_process_rank)

        cname = f'model_out_{partition_id}'
        self.comm.create_channel(cname, self.comm.scheduler_process_rank)

        self.comm.mark_ready()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self.partition_id}WorkerProcess>'


class PrefetchWorker():
    def __init__(self, queue_in: Union[DistributedQueue, mp.Queue], queue_out: LocalQueue,
                 to_cuda: Optional[torch.device] = None):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.to_cuda = to_cuda
        self.running = True
        self.thread = Thread(target=self._thread)
        self.thread.start()

    def _thread(self):
        while self.running:
            try:
                itm = self.queue_in.get(timeout=0.1)
            except Empty:
                continue
            if isinstance(itm, Microbatch):
                traverse_tensor_map(itm.data, lambda t: t.pin_memory() if not t.is_cuda else t)
            if isinstance(itm, Microbatch) and self.to_cuda is not None:
                itm.data = itm.data.to_(self.to_cuda)
            self.queue_out.put(itm)

    def stop(self):
        print("Terminating PrefetchWorker at ", self.thread)
        self.running = False
        self.thread.join()
        print("PrefetchWorker terminated.")


class GpuWorker(BaseWorker):
    def __init__(self,
                 partition_id: Optional[int],
                 num_ubatch: Optional[int],
                 num_bwd_ubatch: Optional[int],
                 communicator_param: CommunicatorParam,
                 optimizer: torch.optim.Optimizer,
                 momentum: float,
                 loss_fn: Callable,
                 update_target_fn: Callable):

        if num_ubatch is None:
            num_ubatch = (communicator_param.world_size - 2)

        if num_bwd_ubatch is None:
            num_bwd_ubatch = num_ubatch * 2

        super().__init__(partition_id, num_ubatch, num_bwd_ubatch, communicator_param, [])

        self.comm: Optional[Communicator] = None
        self.worker_compute = SubGpuWorker(self, 'compute')
        self.worker_copy_batch = SubGpuWorker(self, 'copy_batch')
        self.worker_copy_batch_out = SubGpuWorker(self, 'copy_batch_out')
        self.worker_copy_grad_out = SubGpuWorker(self, 'copy_grad_out')
        self.worker_copy_model_online = SubGpuWorker(self, 'copy_model_online')
        self.worker_copy_model_target = SubGpuWorker(self, 'copy_model_target')
        self.worker_copy_cpu = SubGpuWorker(self, 'copy_cpu')

        self.optimizer = optimizer
        self.momentum = momentum
        self.loss_fn = loss_fn
        self.update_target_fn = update_target_fn

        self.args = None
        self.extra_args = None
        self.partition_online = None
        self.partition_target = None
        
        # Failover 관련 속성 추가
        self.health_check_interval = 10  # 헬스체크 간격 (초)
        self.last_heartbeat = time()
        self.is_failed = False
        self.health_thread: Optional[threading.Thread] = None
        self.health_check_enabled = True
        self.optimizer_pg_param_map: Optional[Dict[int, List[int]]] = None
        # Worker-side slowdown injection config
        self.slowdown_gpu: Optional[int] = None
        self.slowdown_mode: str = "ratio"
        self.slowdown_factor: Optional[float] = None
        self.slowdown_fixed_ms: Optional[float] = None
        self.slowdown_start: Optional[int] = None
        self.slowdown_end: Optional[int] = None
        self.slowdown_warmup_sec: Optional[float] = None
        self.slowdown_duration_sec: Optional[float] = None
        self.slowdown_task_scope: str = "compute"
        self._slowdown_wallclock_anchor_sec: Optional[float] = None
        self._slowdown_batch_start_sec: Dict[int, float] = {}

        # Per-task baseline runtime (ms), learned outside slowdown window
        self._task_baseline_ms: Dict[str, float] = {}
        self._task_baseline_count: Dict[str, int] = defaultdict(int)

        # Logging throttle
        self._slowdown_inject_count: int = 0
        self._slowdown_log_every: int = 20
        self._slowdown_skip_log_every: int = 20
        self._slowdown_skip_counts: Dict[str, int] = defaultdict(int)
        self.start()

    def schedule(self, gpu_task: 'GpuTask'):
        if gpu_task.task_type == TaskType.TASK_COPY_BATCH or gpu_task.task_type == TaskType.TASK_COPY_GRAD:
            return self.worker_copy_batch.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_BATCH_OUT:
            return self.worker_copy_batch_out.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_GRAD_OUT:
            return self.worker_copy_grad_out.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_MODEL:
            return self.worker_copy_model_target.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COMPUTE_FORWARD or gpu_task.task_type == TaskType.TASK_COMPUTE_LOSS:
            return self.worker_compute.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COMPUTE_BACKWARD:
            return self.worker_compute.schedule(gpu_task)  # do this on compute thread too
        elif gpu_task.task_type == TaskType.TASK_COMPUTE_OPTIMIZE_GPU:
            return self.worker_compute.schedule(gpu_task)  # do this on compute thread too
        elif gpu_task.task_type == TaskType.TASK_TERMINATE:
            self.worker_compute.schedule(deepcopy(gpu_task))
            self.worker_copy_cpu.schedule(deepcopy(gpu_task))
            self.worker_copy_batch.schedule(deepcopy(gpu_task))
            self.worker_copy_batch_out.schedule(deepcopy(gpu_task))
            self.worker_copy_grad_out.schedule(deepcopy(gpu_task))
            self.worker_copy_model_online.schedule(deepcopy(gpu_task))
            self.worker_copy_model_target.schedule(deepcopy(gpu_task))
            return
        assert False

    def init_ctx(self):

        rank = self.comm.rank
        self.args: Dict = self.comm.recv(f'init_config_{rank}')
        self.config: Dict = self.comm.recv(f'init_config_{rank}')
        self.extra_args: Dict = self.comm.recv(f'init_config_{rank}')
        self.optimizer_pg_param_map = self.comm.recv(f'init_config_{rank}')
        # Worker-side slowdown config (from train_kd CLI args)
        def _opt(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # user CLI args are carried in extra_args
        arg_src = self.extra_args if self.extra_args is not None else self.args

        self.slowdown_gpu = _opt(arg_src, "inject_slowdown_gpu", None)
        self.slowdown_mode = str(_opt(arg_src, "slowdown_mode", "ratio") or "ratio").strip().lower()
        if self.slowdown_mode not in {"ratio", "fixed"}:
            Log.i(f"[SlowdownConfig] Invalid slowdown_mode={self.slowdown_mode}; fallback to ratio")
            self.slowdown_mode = "ratio"
        self.slowdown_factor = _opt(arg_src, "slowdown_factor", None)
        self.slowdown_fixed_ms = _opt(arg_src, "slowdown_fixed_ms", None)
        self.slowdown_start = _opt(arg_src, "slowdown_start", None)
        self.slowdown_end = _opt(arg_src, "slowdown_end", None)
        self.slowdown_warmup_sec = _opt(arg_src, "slowdown_warmup_sec", None)
        self.slowdown_duration_sec = _opt(arg_src, "slowdown_duration_sec", None)
        self.resume_step_offset = int(_opt(arg_src, "resume_step_offset", 0) or 0)
        self.slowdown_task_scope = _opt(arg_src, "slowdown_task_scope", "compute")
        self.failover_inject_scenario = str(_opt(arg_src, "failover_inject_scenario", "") or "").strip().lower()
        if self.slowdown_task_scope is None:
            self.slowdown_task_scope = "compute"
        self.slowdown_task_scope = str(self.slowdown_task_scope).strip().lower()

        if self.slowdown_warmup_sec is not None:
            self.slowdown_warmup_sec = float(self.slowdown_warmup_sec)
        if self.slowdown_duration_sec is not None:
            self.slowdown_duration_sec = float(self.slowdown_duration_sec)
        if self.slowdown_fixed_ms is not None:
            self.slowdown_fixed_ms = float(self.slowdown_fixed_ms)

        window_desc = f"step_range=({self.slowdown_start}, {self.slowdown_end})"
        if self.slowdown_warmup_sec is not None or self.slowdown_duration_sec is not None:
            window_desc = (
                f"wallclock=(warmup={self.slowdown_warmup_sec}, "
                f"duration={self.slowdown_duration_sec})"
            )

        Log.i(
            f"[SlowdownConfig] partition={self.partition_id}, "
            f"device_id={self.device_id}, target_gpu={self.slowdown_gpu}, "
            f"mode={self.slowdown_mode}, factor={self.slowdown_factor}, fixed_ms={self.slowdown_fixed_ms}, {window_desc}, "
            f"scope={self.slowdown_task_scope}, resume_step_offset={self.resume_step_offset}, "
            f"scenario={self.failover_inject_scenario}, arg_src={type(arg_src).__name__}"
        )       
        Log.d("Received args, args = ", self.args)
        Log.d("Received config, args = ", self.config)

        partition_online, partition_target = self.comm.recv(f'init_model_{rank}')
        self.partition_online = partition_online.to(self.device)
        self.partition_target = partition_target.to(self.device)

        rank, num_ubatch, num_partition = self.partition_id, self.num_ubatch, self.num_partition

        if len(self.list_context) == 0:
            self.list_context = [GpuTaskContext() for _ in range(max(num_ubatch, self.num_bwd_ubatch))]

        params_online, params_target = ParamStorage(), ParamStorage()
        for ctx in self.list_context:
            ctx.device_id = BaseWorker.partition_id_to_device_id(rank)
            ctx.params_online = params_online
            ctx.params_target = params_target
            ctx.config = self.config
            if ctx.config['gpipe_emulation']['enabled']:
                ctx.config['optimizer']['num_skip_initial_staleness'] = None
                print("======= GPipe Emulation Mode Enabled ========")

        params_online.push(0, self.partition_online.parameters())
        params_target.push(0, self.partition_target.parameters())

        for ubatch_id, ctx in enumerate(self.list_context):
            ctx.num_partitions = num_partition
            ctx.num_ubatch = num_ubatch
            ctx.num_bwd_ubatch = self.num_bwd_ubatch
            ctx.args = self.args
            ctx.extra_args = self.extra_args
            ctx.comm = self.comm

            if rank > 0:
                assert ctx.queue_copy_curr_in is None
                if ubatch_id < num_ubatch:
                    ctx.queue_copy_curr_in = DistributedQueue(rank - 1, rank, f'batch_{rank}_{ubatch_id}')
                ctx.queue_grad_out = DistributedQueue(rank, rank-1, f'grad_{rank-1}_{ubatch_id}')
            else:
                if ubatch_id < num_ubatch:
                    ctx.queue_copy_curr_in = DistributedQueue(self.communicator_param.world_size - 1,
                                                              rank, f'batch_{rank}_{ubatch_id}')

            ctx.queue_compute_out = LocalQueue()
            if rank < num_partition - 1:
                if ubatch_id < num_ubatch:
                    ctx.queue_copy_curr_out = DistributedQueue(rank, rank + 1, f'batch_{rank+1}_{ubatch_id}')
                ctx.queue_grad_in = DistributedQueue(rank+1, rank, f'grad_{rank}_{ubatch_id}')

            ctx.partition_online = partition_online
            ctx.partition_target = partition_target

            if rank == num_partition - 1:
                ctx.queue_loss_out = DistributedQueue(rank, self.comm.scheduler_process_rank, 'loss_out')
                if ubatch_id == ctx.num_bwd_ubatch - 1:
                    ctx.queue_label_feed_in = DistributedQueue(rank, self.comm.scheduler_process_rank, 'label_feed')

            ctx.cb_partition_id_to_internal_gpu_id = BaseWorker.partition_id_to_internal_device_id

    def _thread(self):
        Log.v(f"Starting Worker Process for partition {self.partition_id}, device {self.device}")
        self.init_distributed_comm()
        remote_profile_init(DistributedQueue(self.comm.rank, self.comm.scheduler_process_rank,
                                             f'log_{self.partition_id}'))
        self.init_ctx()

        init_gpu_task_profiler(
            output_dir=f"{self.config['__artifact_dir']}/profiling_logs",
            filename=f"gpu_task_summary_partition{self.partition_id}.txt"
        )
        # update optimizer
        try:
            list_param = list(self.partition_online.parameters())
            for pg_id, param_id_lst in self.optimizer_pg_param_map.items():
                self.optimizer.param_groups[pg_id]['params'].clear()
                for param_id in param_id_lst:
                    self.optimizer.param_groups[pg_id]['params'].append(list_param[param_id])

            for ctx in self.list_context:
                ctx.loss_fn = self.loss_fn
                ctx.optimizer = self.optimizer
                ctx.momentum = self.momentum
                ctx.update_target_fn = self.update_target_fn

            self.thread_main()

        finally:
            stop_gpu_task_profiler()

    def thread_main(self):
        device_state = DeviceState(self.partition_id)
        device_id = BaseWorker.partition_id_to_device_id(self.partition_id)
        internal_device_id = BaseWorker.gpu_device_id_to_internal_device_id(device_id)
        cuda_device = torch.device('cuda', internal_device_id)
        prefetch_workers: List[PrefetchWorker] = []

        Log.v(f"Initializing Stream for {self.device}")
        streams = [StreamDescriptor(self.partition_id, StreamType.STREAM_DEFAULT_COMPUTE),
                   StreamDescriptor(self.partition_id, StreamType.STREAM_COPY_BATCH_TO),
                   StreamDescriptor(self.partition_id, StreamType.STREAM_COPY_BATCH_FROM),
                   StreamDescriptor(self.partition_id, StreamType.STREAM_COPY_CPU_TX)]

        def get_stream(desc: StreamDescriptor):
            assert desc.device_id == self.partition_id
            return new_stream(self.device)

        stream_map = {k: get_stream(k) for k in streams}

        local_loss_q = LocalQueue()
        local_grad_q = LocalQueue()
        shared_activation = None
        shared_gradient = None
        shared_final_grad_q = LocalQueue()
        for idx, ctx in enumerate(self.list_context):
            ctx.cuda_streams = stream_map
            ctx.worker = self
            ctx.device_state = device_state

            ctx.queue_compute_in = ctx.queue_copy_out = LocalQueue()

            # Prefetch input
            if ctx.queue_copy_curr_in:
                Log.d(f"Prefetching queue_copy_curr_in for ubatch_idx {idx}")
                ctx.queue_copy_curr_in_unprefetched = ctx.queue_copy_curr_in
                ctx.queue_copy_curr_in = LocalQueue()
                prefetch_workers.append(PrefetchWorker(ctx.queue_copy_curr_in_unprefetched, ctx.queue_copy_curr_in))

            if ctx.queue_label_feed_in:
                Log.d("Prefetching queue_label_feed_in")
                ctx.queue_copy_label_in = LocalQueue()
                prefetch_workers.append(PrefetchWorker(ctx.queue_label_feed_in, ctx.queue_copy_label_in, cuda_device))

            if self.partition_id == self.num_partition - 1:
                # make loss_q here
                ctx.queue_compute_out = local_loss_q
                ctx.queue_loss_compute_in = local_loss_q
            ctx.queue_grad_opt_copy = LocalQueue()

            # if self.partition_id > 0:
            if self.partition_id == self.num_partition - 1:
                ctx.queue_grad_copy = shared_final_grad_q
            else:
                ctx.queue_grad_copy = LocalQueue()
            ctx.queue_grad_pending = LocalQueue()

            if shared_activation is None:
                shared_activation = ActivationStorage(ctx.num_ubatch, f'activation{self.partition_id}')
                shared_gradient = ActivationStorage(ctx.num_ubatch, f'gradient{self.partition_id}')
            ctx.activation = shared_activation
            ctx.gradient = shared_gradient

            if ctx.queue_grad_in is None:
                ctx.queue_grad_in = local_grad_q

            ctx.partition_online, ctx.partition_target = self.partition_online, self.partition_target

        Log.v(f"Initializing Stream for {self.device} Complete!")

        self.worker_level_cv = Condition()

        for w in (self.worker_compute,
                  self.worker_copy_model_online,
                  self.worker_copy_model_target,
                  self.worker_copy_batch,
                  self.worker_copy_batch_out,
                  self.worker_copy_grad_out,
                  self.worker_copy_cpu):
            w.start()

        self.oob_resp_queue.put('ready')
        Log.v(f"Starting Task receiving task_{self.communicator_param.rank}")
        task_queue = DistributedQueue(self.comm.scheduler_process_rank, self.communicator_param.rank,
                                      f'task_{self.communicator_param.rank}')
        while True:
            task: Optional[GpuTask] = task_queue.get()
            # print(f"[Worker-{self.partition_id}] Got task {task}")
            if task is None:
                continue
            # print(f"Got {task}")
            self.schedule(task)
            if task.task_type == TaskType.TASK_TERMINATE:
                print("[LOG] compute_worker_loop received TERMINATE")
                # print(f"[Worker-{self.partition_id}] Received TERMINATE, breaking loop")
                break
        self.cleanup()

        for worker in prefetch_workers:
            worker.stop()

    def cleanup(self):
        print("Cleaning up...")

        self.worker_compute.thread.join()
        print("worker_compute Thread Join OK")
        self.worker_copy_batch.thread.join()
        print("worker_copy_batch Thread Join OK")
        self.worker_copy_batch_out.thread.join()
        print("worker_copy_batch_out Thread Join OK")
        self.worker_copy_grad_out.thread.join()
        print("worker_copy_grad_out Thread Join OK")
        self.worker_copy_model_online.thread.join()
        print("worker_copy_model_online Thread Join OK")
        self.worker_copy_model_target.thread.join()
        print("worker_copy_model_target Thread Join OK")
        self.worker_copy_cpu.thread.join()
        print("worker_copy_cpu Thread Join OK")

        ctx = self.list_context[0]
        model = ctx.partition_target
        trained_model_state_dict = model.cpu().state_dict()

        # Send partitions
        self.comm.send(f'model_out_{self.partition_id}', trained_model_state_dict)

        # dirty
        sleep(1)

        # print("Terminating comm")
        self.comm.finish()

    def start(self):
        self.process = mp.Process(target=self._thread)
        self.process.name = f'GPU{self.device_id}WorkerProcess'
        self.process.start()
        AffinityManager(self.process).set_affinity_for_gpu(self.device_id)
        
        # 헬스체크 스레드 시작
        if self.health_check_enabled:
            self._start_health_check()

    def _start_health_check(self):
        """워커 헬스체크 스레드 시작"""
        if self.health_thread is not None:
            return
            
        self.health_thread = threading.Thread(
            target=self._health_check_loop, 
            daemon=True,
            name=f"HealthCheck-Partition{self.partition_id}"
        )
        self.health_thread.start()
        Log.d(f"Health check started for partition {self.partition_id}")

    def _health_check_loop(self):
        """워커 프로세스 헬스체크 루프"""
        while self.health_check_enabled and not self.is_failed:
            try:
                # 프로세스 생존 확인
                if not self.process.is_alive():
                    Log.e(f"Worker process for partition {self.partition_id} is dead")
                    self.is_failed = True
                    break
                
                # GPU 메모리 접근 테스트 (프로세스 내부에서 실행)
                if not self._test_gpu_accessibility():
                    Log.e(f"GPU accessibility test failed for partition {self.partition_id}")
                    self.is_failed = True
                    break
                    
                # 헬스체크 성공
                self.last_heartbeat = time()
                
                sleep(self.health_check_interval)
                
            except Exception as e:
                Log.e(f"Health check error for partition {self.partition_id}: {e}")
                Log.e(traceback.format_exc())
                self.is_failed = True
                break
        
        if self.is_failed:
            Log.e(f"Worker {self.partition_id} marked as failed")

    def _test_gpu_accessibility(self) -> bool:
        """GPU 접근성 테스트 (간접적)"""
        try:
            # 프로세스가 살아있고, 최근에 heartbeat가 있다면 정상으로 간주
            time_since_heartbeat = time() - self.last_heartbeat
            return time_since_heartbeat < self.health_check_interval * 3
        except Exception:
            return False

    def stop_health_check(self):
        """헬스체크 중지"""
        self.health_check_enabled = False
        if self.health_thread and self.health_thread.is_alive():
            self.health_thread.join(timeout=5)
        
    def get_health_status(self) -> Dict:
        """헬스 상태 반환"""
        return {
            'partition_id': self.partition_id,
            'is_failed': self.is_failed,
            'last_heartbeat': self.last_heartbeat,
            'process_alive': self.process.is_alive() if hasattr(self, 'process') else False,
            'health_check_enabled': self.health_check_enabled
        }
    def _task_scope_of(self, task: 'GpuTask') -> Optional[str]:
        if task.task_type in {
            TaskType.TASK_COMPUTE_FORWARD,
            TaskType.TASK_COMPUTE_BACKWARD,
            TaskType.TASK_COMPUTE_LOSS,
            TaskType.TASK_COMPUTE_OPTIMIZE_GPU,
        }:
            return "compute"

        if task.task_type in {
            TaskType.TASK_COPY_BATCH,
            TaskType.TASK_COPY_BATCH_OUT,
            TaskType.TASK_COPY_GRAD,
            TaskType.TASK_COPY_GRAD_OUT,
            TaskType.TASK_COPY_MODEL,
        }:
            return "comm"

        return None

    def _task_baseline_key(self, task: 'GpuTask') -> Optional[str]:
        scope = self._task_scope_of(task)
        if scope is None:
            return None
        return f"{scope}:{task.task_type.name}"

    def _scope_enabled_for_task(self, task: 'GpuTask') -> bool:
        scope = self._task_scope_of(task)
        if scope is None:
            return False

        if self.slowdown_task_scope == "both":
            return True
        if self.slowdown_task_scope == "compute" and scope == "compute":
            return True
        if self.slowdown_task_scope == "comm" and scope == "comm":
            return True
        return False

    def _is_target_slowdown_gpu(self) -> bool:
        if self.slowdown_gpu is None:
            return True
        if self.device_id is None:
            return False
        return int(self.device_id) == int(self.slowdown_gpu)

    def _slowdown_step_key(self, task: 'GpuTask') -> int:
        return int(self._task_global_step(task))

    def _record_slowdown_batch_start(self, task: 'GpuTask') -> float:
        batch_key = self._slowdown_step_key(task)
        start_sec = self._slowdown_batch_start_sec.get(batch_key)
        if start_sec is None:
            start_sec = monotonic()
            self._slowdown_batch_start_sec[batch_key] = start_sec
        return start_sec

    def _consume_slowdown_batch_start(self, task: 'GpuTask') -> Optional[float]:
        return self._slowdown_batch_start_sec.pop(self._slowdown_step_key(task), None)

    def _task_local_step(self, task: 'GpuTask') -> int:
        return max(0, int(task.batch_id) - 1)

    def _task_global_step(self, task: 'GpuTask') -> int:
        return int(self.resume_step_offset) + self._task_local_step(task)

    def _has_wallclock_slowdown_window(self) -> bool:
        return (
            self.slowdown_warmup_sec is not None and
            self.slowdown_duration_sec is not None
        )

    def _ensure_slowdown_wallclock_anchor(self, task: 'GpuTask') -> float:
        if self._slowdown_wallclock_anchor_sec is None:
            self._slowdown_wallclock_anchor_sec = monotonic()
            Log.i(
                f"⏱️ [SlowdownAnchor] partition={self.partition_id}, "
                f"device_id={self.device_id}, batch={task.batch_id}, "
                f"warmup_sec={self.slowdown_warmup_sec}, "
                f"duration_sec={self.slowdown_duration_sec}"
            )
        return self._slowdown_wallclock_anchor_sec

    def _capture_task_slowdown_window_state(self, task: 'GpuTask') -> dict:
        cached = getattr(task, "_worker_slowdown_window_state", None)
        if cached is not None:
            return cached

        if self._has_wallclock_slowdown_window():
            anchor_sec = self._ensure_slowdown_wallclock_anchor(task)
            elapsed_sec = max(0.0, monotonic() - anchor_sec)
            start_sec = float(self.slowdown_warmup_sec)
            end_sec = start_sec + float(self.slowdown_duration_sec)
            state = {
                "mode": "wallclock",
                "in_window": start_sec <= elapsed_sec < end_sec,
                "elapsed_sec": elapsed_sec,
                "window_start_sec": start_sec,
                "window_end_sec": end_sec,
                "label": (
                    f"warmup={start_sec:.2f}s, duration={float(self.slowdown_duration_sec):.2f}s, "
                    f"elapsed={elapsed_sec:.2f}s"
                ),
            }
        elif self.slowdown_start is not None and self.slowdown_end is not None:
            global_step = self._task_global_step(task)
            state = {
                "mode": "step",
                "in_window": int(self.slowdown_start) <= global_step < int(self.slowdown_end),
                "global_step": global_step,
                "window_start_step": int(self.slowdown_start),
                "window_end_step": int(self.slowdown_end),
                "label": f"range=({self.slowdown_start}, {self.slowdown_end}), global_step={global_step}",
            }
        else:
            state = {
                "mode": "none",
                "in_window": False,
                "label": "no_window",
            }

        setattr(task, "_worker_slowdown_window_state", state)
        return state

    def _is_in_slowdown_window(self, task: 'GpuTask') -> bool:
        return bool(self._capture_task_slowdown_window_state(task).get("in_window", False))

    def _log_slowdown_skip(self, task: 'GpuTask', reason: str, extra: Optional[str] = None):
        key = self._task_baseline_key(task) or "unknown"
        scope = self._task_scope_of(task)
        local_step = self._task_local_step(task)
        global_step = self._task_global_step(task)
        window_state = self._capture_task_slowdown_window_state(task)
        counter_key = f"{reason}:{key}"
        self._slowdown_skip_counts[counter_key] += 1
        if self._slowdown_skip_counts[counter_key] % self._slowdown_skip_log_every != 0:
            return

        suffix = f", {extra}" if extra else ""
        Log.i(
            f"🛑 [WorkerSlowdownSkip] partition={self.partition_id}, device_id={self.device_id}, "
            f"task={task.task_type.name}, scope={scope}, batch={task.batch_id}, "
            f"local_step={local_step}, global_step={global_step}, "
            f"window={window_state.get('label')}, reason={reason}{suffix}"
        )

    def should_inject_task_slowdown(self, task: 'GpuTask') -> bool:
        if self.slowdown_mode == "fixed":
            if self.slowdown_fixed_ms is None:
                self._log_slowdown_skip(task, "missing_fixed_ms")
                return False
            if float(self.slowdown_fixed_ms) <= 0.0:
                self._log_slowdown_skip(task, "non_positive_fixed_ms", f"fixed_ms={self.slowdown_fixed_ms}")
                return False
        else:
            if self.slowdown_factor is None:
                self._log_slowdown_skip(task, "missing_factor")
                return False
            if float(self.slowdown_factor) <= 1.0:
                self._log_slowdown_skip(task, "non_positive_factor", f"factor={self.slowdown_factor}")
                return False
        if not self._is_target_slowdown_gpu():
            self._log_slowdown_skip(
                task,
                "gpu_mismatch",
                f"target_gpu={self.slowdown_gpu}"
            )
            return False
        if not self._scope_enabled_for_task(task):
            self._log_slowdown_skip(
                task,
                "scope_mismatch",
                f"configured_scope={self.slowdown_task_scope}"
            )
            return False
        window_state = self._capture_task_slowdown_window_state(task)
        if not window_state.get("in_window", False):
            self._log_slowdown_skip(
                task,
                "outside_window",
                window_state.get("label")
            )
            return False
        return True

    def get_task_injected_sleep_sec(self, task: 'GpuTask') -> float:
        setattr(task, "_worker_slowdown_injection_applied", False)

        if self.failover_inject_scenario == "slowdown":
            if not self.should_inject_task_slowdown(task):
                return 0.0

            # Warm-up interval 동안은 기준 시각만 기록하고 실제 sleep은 하지 않는다.
            self._record_slowdown_batch_start(task)

            if task.task_type != TaskType.TASK_COMPUTE_OPTIMIZE_GPU:
                return 0.0

            ratio = float(self.slowdown_factor or 1.0)
            if ratio <= 1.0:
                self._log_slowdown_skip(task, "non_positive_ratio", f"ratio={ratio}")
                self._consume_slowdown_batch_start(task)
                return 0.0

            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            start_sec = self._consume_slowdown_batch_start(task)
            if start_sec is None:
                start_sec = monotonic()

            elapsed_sec = max(monotonic() - start_sec, 0.0)
            extra_sec = max(elapsed_sec * (ratio - 1.0), 0.0)
            if extra_sec <= 0.0:
                return 0.0

            setattr(task, "_worker_slowdown_injection_applied", True)
            return extra_sec

        if not self.should_inject_task_slowdown(task):
            return 0.0

        if self.slowdown_mode == "fixed":
            fixed_ms = float(self.slowdown_fixed_ms or 0.0)
            if fixed_ms <= 0.0:
                self._log_slowdown_skip(task, "zero_fixed_sleep", f"fixed_ms={fixed_ms:.2f}")
                return 0.0
            setattr(task, "_worker_slowdown_injection_applied", True)
            return fixed_ms / 1000.0

        key = self._task_baseline_key(task)
        if key is None:
            self._log_slowdown_skip(task, "missing_baseline_key")
            return 0.0

        base_ms = self._task_baseline_ms.get(key)
        if base_ms is None or base_ms <= 0:
            baseline_count = int(self._task_baseline_count.get(key, 0))
            self._log_slowdown_skip(
                task,
                "missing_baseline",
                f"baseline_key={key}, baseline_ms={base_ms}, baseline_count={baseline_count}"
            )
            return 0.0

        factor = float(self.slowdown_factor)
        extra_ms = max(base_ms * (factor - 1.0), 0.0)
        if extra_ms <= 0:
            self._log_slowdown_skip(
                task,
                "zero_extra_sleep",
                f"baseline_key={key}, baseline_ms={base_ms:.2f}, factor={factor:.2f}"
            )
            return 0.0
        setattr(task, "_worker_slowdown_injection_applied", True)
        return extra_ms / 1000.0

    def update_task_baseline(self, task: 'GpuTask', elapsed_ms: float):
        key = self._task_baseline_key(task)
        if key is None:
            return
        if elapsed_ms <= 0:
            return

        # 주입 구간 값으로 baseline 오염 방지
        if self._is_target_slowdown_gpu() and self._is_in_slowdown_window(task):
            return

        prev = self._task_baseline_ms.get(key)
        if prev is None:
            self._task_baseline_ms[key] = float(elapsed_ms)
        else:
            self._task_baseline_ms[key] = 0.9 * prev + 0.1 * float(elapsed_ms)

        self._task_baseline_count[key] += 1
        if self._is_target_slowdown_gpu() and self._task_baseline_count[key] in {1, 5, 20}:
            Log.i(
                f"📏 [WorkerSlowdownBaseline] partition={self.partition_id}, device_id={self.device_id}, "
                f"key={key}, count={self._task_baseline_count[key]}, "
                f"baseline_ms={self._task_baseline_ms[key]:.2f}"
            )

    def log_task_slowdown_injection(self, task: 'GpuTask', sleep_sec: float):
        self._slowdown_inject_count += 1
        if self.failover_inject_scenario != "slowdown" and self._slowdown_inject_count % self._slowdown_log_every != 0:
            return

        Log.i(
            f"[Injected Delay] GPU {self.device_id}: Sleeping {sleep_sec:.2f}s "
            f"(partition={self.partition_id}, task={task.task_type.name}, batch={task.batch_id}, "
            f"local_step={self._task_local_step(task)}, global_step={self._task_global_step(task)}, "
            f"ratio={self.slowdown_factor}, scenario={self.failover_inject_scenario})"
        )
    def join(self):
        # 헬스체크 중지
        self.stop_health_check()
        self.process.join()


class LocalWorker():
    def __init__(self, lst_local_ctx: List['LocalTaskContext']):
        self.device = None
        self.list_context = lst_local_ctx
        self.worker_compute = SubCpuWorker(self, 'cmpt')
        self.worker_copy_batch = SubCpuWorker(self, 'cpyb')
        self.start()

    def start(self):
        Log.v(f"Initializing Localworker {self}")

        self.worker_level_cv = Condition()
        self.worker_compute.start()
        self.worker_copy_batch.start()

    def schedule(self, gpu_task: 'GpuTask'):
        if gpu_task.task_type == TaskType.TASK_FEED_BATCH:
            self.worker_copy_batch.schedule(gpu_task)
        else:
            self.worker_compute.schedule(gpu_task)

    def stop(self):
        print("Stopping LocalWorker")
        self.worker_compute.stop()
        self.worker_copy_batch.stop()

        print("Waiting for LocalWorker join")
        
        self.worker_compute.thread.join()
        self.worker_copy_batch.thread.join()
        
        print(f"[DEBUG] {self.worker_compute.thread.name} alive: {self.worker_compute.thread.is_alive()}")
        print(f"[DEBUG] {self.worker_copy_batch.thread.name} alive: {self.worker_copy_batch.thread.is_alive()}")
        print("LocalWorker stopped.")
