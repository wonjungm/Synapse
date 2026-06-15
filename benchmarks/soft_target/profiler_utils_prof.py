
from typing import Optional
from pathlib import Path
from threading import Thread
from queue import Queue
from collections import defaultdict, deque
import pynvml as nvml
import time
import threading
import torch.cuda.nvtx as nvtx
import torch

class GpuTaskProfiler:
    def __init__(self, output_dir: str = "profiling_logs", filename: str = "gpu_task_summary.txt"):
        self.queue = Queue()
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / filename
        self.records = []
        self.running = True
        self.thread = Thread(target=self._thread_logger, name="ProfilerThread", daemon=True)
        self.thread.start()
        print(f"[Profiler] Logging to {filename}")

    def log(self, task_name, device_id, batch_id, ubatch_id, partition_id, is_target, time_ms, mem, max_mem):
        self.queue.put({
            'task_name': task_name,
            'device': device_id,
            'batch_id': batch_id,
            'ubatch_id': ubatch_id,
            'partition': partition_id,
            'target': is_target,
            'time_ms': time_ms,
            'mem_MB': mem,
            'max_mem_MB': max_mem
        })
    
    def _thread_logger(self):
        while self.running or not self.queue.empty():
            try:
                record = self.queue.get()
                if record == "STOP":
                    break
                self.records.append(record)
            except:
                continue
        self._save_summary()

    def stop(self):
        self.running = False
        self.queue.put("STOP")
        self.thread.join()

    def _save_summary(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            by_step = defaultdict(list)
            for record in self.records:
                by_step[record['batch_id']].append(record)

            for step, records in sorted(by_step.items()):
                f.write(f"\n====== Step {step} ======\n")
                for record in records:
                    f.write(f"[GPU {record['device']}] Task={record['task_name']:<20} | "
                            f"Batch={record['batch_id']:<3} UBatch={record['ubatch_id']} "
                            f"Partition={record['partition']} Target={record['target']} | "
                            f"Time={record['time_ms']:.2f} ms | Mem={record['mem_MB']:.2f} MB | MaxMem={record['max_mem_MB']:.2f} MB\n")
        print(f"Profiling summary saved to {self.output_file}")

gpu_task_profiler_instance: Optional[GpuTaskProfiler] = None

def init_gpu_task_profiler(*args, **kwargs):
    global gpu_task_profiler_instance
    gpu_task_profiler_instance = GpuTaskProfiler(*args, **kwargs)

def stop_gpu_task_profiler():
    global gpu_task_profiler_instance
    if gpu_task_profiler_instance is not None:
        gpu_task_profiler_instance.stop()

def create_compute_profile_hooks(task_name, task_fn):
    def wrapped_fn(ctx, task):
        # === NVTX tracing condition ===
        do_nvtx_trace = task.batch_id is not None and 6 <= task.batch_id <= 10
        if do_nvtx_trace:
            label = (
                f"Task={task_name:<20} | "
                f"Batch={task.batch_id:<3} UBatch={task.ubatch_id:<2} | "
                f"Partition={task.partition_id} Target={task.is_target} | "
                f"GPU={ctx.device_id}"
            )
            nvtx.range_push(label)
        
        # === Timing setup ===
        # torch.cuda.synchronize()
        # start_time = time.time()
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        try:
            result = task_fn(ctx, task)
        except Exception as e:
            print(f"[GPU {ctx.device_id}] Exception during task {task_name}: {e}")
            if do_nvtx_trace:
                nvtx.range_pop()
            raise
        if do_nvtx_trace:
            nvtx.range_pop()

        # end.record()
        # torch.cuda.synchronize()
        # end_time = time.time()

        # if task.batch_id >= 6 and task.batch_id <= 10:
        #     print(f'{task_name},{task.batch_id},{task.ubatch_id},{task.partition_id},{task.is_target},{ctx.device_id}')
        return result
    return wrapped_fn

class GpuUtilSampler:
    def __init__(self, interval=0.01, maxlen=5000):
        self.interval = interval
        self.running = False
        self.data = deque(maxlen=maxlen)  # (timestamp, device_id, util, power)
        self.thread = threading.Thread(target=self._run, daemon=True)
        nvml.nvmlInit()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]

    def _run(self):
        while self.running:
            timestamp = time.time()
            for i, handle in enumerate(self.handles):
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                self.data.append((timestamp, i, util.gpu, power))
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_average_util(self, device_id, start, end):
        samples = [u for t, i, u, _ in self.data if i == device_id and start <= t <= end]
        return sum(samples) / len(samples) if samples else 0.0

    def get_average_power(self, device_id, start, end):
        samples = [p for t, i, _, p in self.data if i == device_id and start <= t <= end]
        return sum(samples) / len(samples) if samples else 0.0

# Global instance
# gpu_util_sampler = GpuUtilSampler()
# gpu_util_sampler.start()