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

    def log(self, task_name, device_id, batch_id, ubatch_id, partition_id, is_target, time_ms, mem, max_mem, start_time=None, gpu_util=None, power_w=None, power_limit_w=None):
        self.queue.put({
            'task_name': task_name,
            'device': device_id,
            'batch_id': batch_id,
            'ubatch_id': ubatch_id,
            'partition': partition_id,
            'target': is_target,
            'time_ms': time_ms,
            'mem_MB': mem,
            'max_mem_MB': max_mem,
            'start_time': start_time,
            'gpu_util': gpu_util,
            'power_w': power_w,
            'power_limit_w': power_limit_w,
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
        with open(self.output_file, "w") as f:
            by_step = defaultdict(list)
            for r in self.records:
                by_step[r['batch_id']].append(r)

            for step, records in sorted(by_step.items()):
                f.write(f"\n====== Step {step} ======\n")
                for r in records:
                    line = (
                        f"[GPU {r.get('device')}] "
                        f"Task={r.get('task_name', ''):<20} | "
                        f"Batch={r.get('batch_id')} "
                        f"UBatch={r.get('ubatch_id')} | "
                        f"Partition={r.get('partition')} "
                        f"Target={r.get('target')} | "
                    )

                    if r.get('start_time') is not None:
                        line += f"StartTime={r['start_time']:.6f} | "

                    line += f"Time={r.get('time_ms', 0.0):.2f} ms | "

                    line += (
                        f"Mem={r.get('mem_MB', 0.0):.2f} MB | "
                        f"MaxMem={r.get('max_mem_MB', 0.0):.2f} MB"
                    )

                    # ---- GPU util ----
                    if r.get('gpu_util') is not None:
                        line += f" | GpuUtil={r['gpu_util']}%"

                    # ---- Power ----
                    if r.get('power_w') is not None:
                        power_limit = r.get('power_limit_w')
                        if power_limit is not None:
                            line += f" | Power={r['power_w']:.2f}/{power_limit:.2f}W"
                        else:
                            line += f" | Power={r['power_w']:.2f}W"

                    f.write(line + "\n")

        print(f"[Profiler] Profiling summary saved to {self.output_file}")


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
        # ========= One-time NVML init =========
        if not hasattr(wrapped_fn, "nvml_initialized"):
            nvml.nvmlInit()
            wrapped_fn.gpu_handles = [
                nvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(nvml.nvmlDeviceGetCount())
            ]
            wrapped_fn.nvml_initialized = True

        # ========= Profiling window =========
        if task.batch_id is None or not (6 <= task.batch_id <= 10):
            return task_fn(ctx, task)

        device_id = ctx.device_id

        # ========= Timing (CPU + CUDA) =========
        torch.cuda.synchronize()
        wall_start = time.time()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        try:
            result = task_fn(ctx, task)
        except Exception as e:
            print(f"[GPU {device_id}] Exception during task {task_name}: {e}")
            raise

        end_evt.record()
        torch.cuda.synchronize()

        elapsed_ms = start_evt.elapsed_time(end_evt)

        # ========= Memory =========
        mem_mb = (
            torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            if device_id is not None else 0.0
        )
        max_mem_mb = (
            torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)
            if device_id is not None else 0.0
        )

        # ========= NVML metrics =========
        gpu_util = None
        power_w = None
        power_limit_w = None

        if device_id is not None and device_id < len(wrapped_fn.gpu_handles):
            handle = wrapped_fn.gpu_handles[device_id]
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            power_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit_w = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

        # ========= TXT logging =========
        if gpu_task_profiler_instance is not None:
            gpu_task_profiler_instance.log(
                task_name=task_name,
                device_id=device_id,
                batch_id=task.batch_id,
                ubatch_id=task.ubatch_id,
                partition_id=task.partition_id,
                is_target=task.is_target,
                time_ms=elapsed_ms,
                mem=mem_mb,
                max_mem=max_mem_mb,
                start_time=wall_start,
                gpu_util=gpu_util,
                power_w=power_w,
                power_limit_w=power_limit_w,
            )

        # ========= Optional console debug =========
        # print(
        #     f"[GPU {device_id}] Task={task_name:<20} "
        #     f"| Batch={task.batch_id:<3} UBatch={task.ubatch_id:<2} "
        #     f"| Part={task.partition_id} Target={task.is_target} "
        #     f"| Time={elapsed_ms:.2f} ms "
        #     f"| Mem={mem_mb:.2f}/{max_mem_mb:.2f} MB "
        #     f"| Util={gpu_util}% Power={power_w:.1f}/{power_limit_w:.1f} W"
        # )

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