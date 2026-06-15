from typing import Optional
from pathlib import Path
from threading import Thread
from queue import Queue
from collections import defaultdict, deque
import json
import os
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
        self.trace_file = self.output_dir / f"{self.output_file.stem}.jsonl"
        self.records = []
        self.running = True
        self.thread = Thread(target=self._thread_logger, name="ProfilerThread", daemon=True)
        self.thread.start()
        print(f"[Profiler] Logging to {filename}")

    def _append_trace_record(self, record):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log(
        self,
        task_name,
        device_id,
        batch_id,
        ubatch_id,
        partition_id,
        is_target,
        time_ms,
        mem,
        max_mem,
        start_time=None,
        gpu_util=None,
        power_w=None,
        power_limit_w=None,
        wall_ms=None,
        cuda_ms=None,
        queue_wait_ms=None,
        sync_wait_ms=None,
        injected_sleep_ms=None,
        exec_wall_ms=None,
        global_step=None,
        global_batch_id=None,
    ):
        record = {
            "task_name": task_name,
            "device": device_id,
            "batch_id": batch_id,
            "global_step": global_step,
            "global_batch_id": global_batch_id,
            "ubatch_id": ubatch_id,
            "partition": partition_id,
            "target": is_target,
            "time_ms": time_ms,              # backward compatibility
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "queue_wait_ms": queue_wait_ms,
            "sync_wait_ms": sync_wait_ms,
            "injected_sleep_ms": injected_sleep_ms,
            "exec_wall_ms": exec_wall_ms,
            "mem_MB": mem,
            "max_mem_MB": max_mem,
            "start_time": start_time,
            "gpu_util": gpu_util,
            "power_w": power_w,
            "power_limit_w": power_limit_w,
        }
        self._append_trace_record(record)
        self.queue.put(record)

    def _thread_logger(self):
        while self.running or not self.queue.empty():
            try:
                record = self.queue.get()
                if record == "STOP":
                    break
                self.records.append(record)
            except Exception:
                continue
        self._save_summary()

    def stop(self):
        self.running = False
        self.queue.put("STOP")
        self.thread.join()

    def _load_trace_records(self):
        if not self.trace_file.exists():
            return list(self.records)

        records = []
        with open(self.trace_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _save_summary(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_records = self._load_trace_records()
        with open(self.output_file, "w") as f:
            by_step = defaultdict(list)
            for r in all_records:
                step_key = r.get("global_batch_id")
                if step_key is None:
                    step_key = r.get("batch_id")
                by_step[step_key].append(r)

            for step, records in sorted(by_step.items()):
                f.write(f"\n====== Step {step} ======\n")
                for r in records:
                    global_batch_id = r.get("global_batch_id")
                    local_batch_id = r.get("batch_id")
                    global_step = r.get("global_step")

                    line = (
                        f"[GPU {r.get('device')}] "
                        f"Task={r.get('task_name', ''):<20} | "
                    )

                    if global_batch_id is not None and global_batch_id != local_batch_id:
                        line += (
                            f"GlobalBatch={global_batch_id} "
                            f"LocalBatch={local_batch_id} "
                        )
                    else:
                        line += f"Batch={local_batch_id} "

                    if global_step is not None:
                        line += f"GlobalStep={global_step} "

                    line += (
                        f"UBatch={r.get('ubatch_id')} | "
                        f"Partition={r.get('partition')} "
                        f"Target={r.get('target')} | "
                    )

                    if r.get("start_time") is not None:
                        line += f"StartTime={r['start_time']:.6f} | "

                    line += (
                        f"Wall={r.get('wall_ms', 0.0):.2f} ms | "
                        f"CUDA={r.get('cuda_ms', 0.0):.2f} ms | "
                        f"ExecWall={r.get('exec_wall_ms', 0.0):.2f} ms | "
                        f"QWait={r.get('queue_wait_ms', 0.0):.2f} ms | "
                        f"SyncWait={r.get('sync_wait_ms', 0.0):.2f} ms | "
                        f"InjectedSleep={r.get('injected_sleep_ms', 0.0):.2f} ms | "
                    )

                    line += (
                        f"Mem={r.get('mem_MB', 0.0):.2f} MB | "
                        f"MaxMem={r.get('max_mem_MB', 0.0):.2f} MB"
                    )

                    if r.get("gpu_util") is not None:
                        line += f" | GpuUtil={r['gpu_util']}%"

                    if r.get("power_w") is not None:
                        power_limit = r.get("power_limit_w")
                        if power_limit is not None:
                            line += f" | Power={r['power_w']:.2f}/{power_limit:.2f}W"
                        else:
                            line += f" | Power={r['power_w']:.2f}W"

                    f.write(line + "\n")

        print(f"[Profiler] Profiling summary saved to {self.output_file}")


gpu_task_profiler_instance: Optional[GpuTaskProfiler] = None
gpu_util_sampler_instance: Optional["GpuUtilSampler"] = None


def init_gpu_task_profiler(*args, **kwargs):
    global gpu_task_profiler_instance, gpu_util_sampler_instance
    gpu_task_profiler_instance = GpuTaskProfiler(*args, **kwargs)
    if gpu_util_sampler_instance is None:
        gpu_util_sampler_instance = GpuUtilSampler()
        gpu_util_sampler_instance.start()


def stop_gpu_task_profiler():
    global gpu_task_profiler_instance, gpu_util_sampler_instance
    if gpu_task_profiler_instance is not None:
        gpu_task_profiler_instance.stop()
        gpu_task_profiler_instance = None
    if gpu_util_sampler_instance is not None:
        gpu_util_sampler_instance.stop()
        gpu_util_sampler_instance = None


def _parse_visible_device_tokens():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return None

    tokens = [token.strip() for token in visible_devices.split(",")]
    return [token for token in tokens if token]


def _resolve_visible_cuda_index(ctx, fallback_device_id):
    worker = getattr(ctx, "worker", None)
    internal_device_id = getattr(worker, "internal_device_id", None)
    if internal_device_id is not None:
        return internal_device_id
    if fallback_device_id is not None:
        return fallback_device_id
    try:
        return torch.cuda.current_device()
    except Exception:
        return None


def _resolve_nvml_device_index(ctx, fallback_device_id):
    visible_cuda_index = _resolve_visible_cuda_index(ctx, fallback_device_id)
    if visible_cuda_index is None:
        return None

    visible_tokens = _parse_visible_device_tokens()
    if visible_tokens is None:
        return visible_cuda_index

    if visible_cuda_index >= len(visible_tokens):
        return None

    token = visible_tokens[visible_cuda_index]
    if token.isdigit():
        return int(token)

    # Avoid silently sampling the wrong GPU when CUDA_VISIBLE_DEVICES uses UUID/MIG tokens.
    return None


def _consume_profile_extra(ctx):
    extra = getattr(ctx, "_profile_extra", None)
    if not isinstance(extra, dict):
        return {
            "queue_wait_ms": 0.0,
            "sync_wait_ms": 0.0,
            "injected_sleep_ms": 0.0,
        }

    out = {
        "queue_wait_ms": float(extra.get("queue_wait_ms", 0.0) or 0.0),
        "sync_wait_ms": float(extra.get("sync_wait_ms", 0.0) or 0.0),
        "injected_sleep_ms": float(extra.get("injected_sleep_ms", 0.0) or 0.0),
    }
    ctx._profile_extra = {}
    return out


def create_compute_profile_hooks(task_name, task_fn, record_gpu_util=True):
    def wrapped_fn(ctx, task):
        if not hasattr(wrapped_fn, "nvml_initialized"):
            nvml.nvmlInit()
            wrapped_fn.gpu_handles = [
                nvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(nvml.nvmlDeviceGetCount())
            ]
            wrapped_fn.nvml_initialized = True

        if task.batch_id is None:
            return task_fn(ctx, task)

        device_id = ctx.device_id

        # profile extras reset
        ctx._profile_extra = {}

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

        wall_end = time.time()
        wall_ms = (wall_end - wall_start) * 1000.0
        cuda_ms = start_evt.elapsed_time(end_evt)

        extra = _consume_profile_extra(ctx)
        queue_wait_ms = extra["queue_wait_ms"]
        sync_wait_ms = extra["sync_wait_ms"]
        injected_sleep_ms = extra["injected_sleep_ms"]

        # localization input용: 전체 wall에서 queue wait만 제외
        # injected_sleep_ms는 synthetic 실험에서 localization에 반영되어야 하므로 포함
        exec_wall_ms = max(wall_ms - queue_wait_ms, 0.0)

        mem_mb = (
            torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            if device_id is not None else 0.0
        )
        max_mem_mb = (
            torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)
            if device_id is not None else 0.0
        )

        gpu_util = None
        power_w = None
        power_limit_w = None
        nvml_device_index = _resolve_nvml_device_index(ctx, device_id)

        if (
            gpu_util_sampler_instance is not None
            and nvml_device_index is not None
        ):
            # Figure-facing GPU utilization should reflect the whole task window,
            # not a single instantaneous NVML sample at task completion.
            if record_gpu_util:
                gpu_util = gpu_util_sampler_instance.get_average_util(
                    nvml_device_index,
                    wall_start,
                    wall_end,
                )
            power_w = gpu_util_sampler_instance.get_average_power(
                nvml_device_index,
                wall_start,
                wall_end,
            )

        if (
            nvml_device_index is not None
            and nvml_device_index < len(wrapped_fn.gpu_handles)
        ):
            handle = wrapped_fn.gpu_handles[nvml_device_index]
            if record_gpu_util and gpu_util is None:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            if power_w is None:
                power_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit_w = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

        global_step = None
        global_batch_id = None
        worker = getattr(ctx, "worker", None)
        if worker is not None and hasattr(worker, "_task_global_step"):
            try:
                global_step = int(worker._task_global_step(task))
                global_batch_id = int(global_step) + 1
            except Exception:
                global_step = None
                global_batch_id = None

        if gpu_task_profiler_instance is not None:
            gpu_task_profiler_instance.log(
                task_name=task_name,
                device_id=device_id,
                batch_id=task.batch_id,
                ubatch_id=task.ubatch_id,
                partition_id=task.partition_id,
                is_target=task.is_target,
                time_ms=exec_wall_ms,    # backward compatibility: old readers still see useful value
                wall_ms=wall_ms,
                cuda_ms=cuda_ms,
                queue_wait_ms=queue_wait_ms,
                sync_wait_ms=sync_wait_ms,
                injected_sleep_ms=injected_sleep_ms,
                exec_wall_ms=exec_wall_ms,
                mem=mem_mb,
                max_mem=max_mem_mb,
                start_time=wall_start,
                gpu_util=gpu_util,
                power_w=power_w,
                power_limit_w=power_limit_w,
                global_step=global_step,
                global_batch_id=global_batch_id,
            )

        return result

    return wrapped_fn


class GpuUtilSampler:
    def __init__(self, interval=0.01, maxlen=5000):
        self.interval = interval
        self.running = False
        self.data = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.thread = None
        nvml.nvmlInit()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]

    def _run(self):
        while self.running:
            timestamp = time.time()
            samples = []
            for i, handle in enumerate(self.handles):
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                samples.append((timestamp, i, util.gpu, power))
            with self.lock:
                self.data.extend(samples)
            time.sleep(self.interval)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def get_average_util(self, device_id, start, end):
        with self.lock:
            snapshot = list(self.data)
        samples = [u for t, i, u, _ in snapshot if i == device_id and start <= t <= end]
        return sum(samples) / len(samples) if samples else None

    def get_average_power(self, device_id, start, end):
        with self.lock:
            snapshot = list(self.data)
        samples = [p for t, i, _, p in snapshot if i == device_id and start <= t <= end]
        return sum(samples) / len(samples) if samples else None
