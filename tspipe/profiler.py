from threading import Thread
from time import sleep, time_ns
from typing import Optional
import json
from datetime import datetime

from tspipe.communicator import DistributedQueue
from tspipe.logger import Log
from tspipe.multiprocessing import Queue

__all__ = ['TSPipeProfiler', 'profile_semantic', 'profile_init']


def timestamp():
    return time_ns() // 1_000_000


class TSPipeProfilerContainer:
    def __init__(self, profiler: Optional['TSPipeProfiler'] = None):
        self.profiler = profiler


current_profiler_container: Optional['TSPipeProfilerContainer'] = TSPipeProfilerContainer()
is_master_process = False


class TSPipeProfiler:
    def __init__(self, filename):
        # self.lock = Lock()
        self.filename = filename
        self.running = True
        self.profiler_message_queue = Queue()
        self.f = None
        self.thd = None
        
        # Failover 관련 속성 추가
        self.last_known_good_config = None
        self.failover_mode = False
        self.failover_events = []
        self.restart_count = 0

    def __enter__(self):
        global is_master_process
        current_profiler_container.profiler = self
        print("Profiler Activated.")
        is_master_process = True
        
        # 프로파일링 시작 이벤트 기록
        self._log_profiler_event("profiler_start", {"filename": self.filename})

    def __exit__(self, type, value, trace_back):
        # 프로파일링 종료 이벤트 기록
        self._log_profiler_event("profiler_stop")
        
        if self.f is not None:
            self.f.close()
        current_profiler_container.profiler = None
        self.running = False
        if self.thd is not None:
            self.thd.join()

    def profile_semantic(self, ts, batch_idx, view_idx, ubatch_idx, is_target, src_partition, dst_partition, op_type):
        return ",".join(str(t) for t in [
            ts, batch_idx, view_idx, ubatch_idx, is_target, src_partition, dst_partition, op_type])

    def restart_with_new_config(self, new_partition_config, reason="gpu_failure"):
        """새로운 GPU 구성으로 프로파일러 재시작"""
        Log.i(f"🔄 Restarting profiler due to: {reason}")
        
        self.failover_mode = True
        self.restart_count += 1
        
        # 현재 상태 저장
        self._save_current_state(reason, new_partition_config)
        
        # 프로파일링 큐 클리어 (옵션)
        try:
            while not self.profiler_message_queue.empty():
                self.profiler_message_queue.get_nowait()
        except:
            pass
        
        # 새로운 구성으로 재시작 이벤트 기록
        self._log_profiler_event("profiler_restart", {
            "reason": reason,
            "old_config": self.last_known_good_config,
            "new_config": new_partition_config,
            "restart_count": self.restart_count
        })
        
        # 새로운 구성 저장
        self.last_known_good_config = new_partition_config
        
        Log.i("✅ Profiler restarted successfully")

    def _save_current_state(self, reason, new_config):
        """현재 프로파일링 상태 저장"""
        if self.f:
            # 파일에 failover 이벤트 마커 추가
            failover_marker = f"# FAILOVER_EVENT: {reason} at {datetime.now().isoformat()}\n"
            failover_marker += f"# OLD_CONFIG: {json.dumps(self.last_known_good_config)}\n"
            failover_marker += f"# NEW_CONFIG: {json.dumps(new_config)}\n"
            failover_marker += f"# RESTART_COUNT: {self.restart_count + 1}\n"
            
            self.f.write(failover_marker)
            self.f.flush()

    def handle_gpu_failure(self, failed_gpu_id, affected_partitions):
        """GPU 실패 처리"""
        self._log_profiler_event("gpu_failure", {
            "failed_gpu_id": failed_gpu_id,
            "affected_partitions": affected_partitions,
            "timestamp": datetime.now().isoformat()
        })
        
        # GPU 실패로 인한 프로파일링 재구성
        if self.f:
            self.f.write(f"# GPU_FAILURE: GPU {failed_gpu_id} failed, partitions {affected_partitions} affected\n")
            self.f.flush()

    def handle_recovery_complete(self, recovery_time_ms, new_config):
        """복구 완료 처리"""
        self._log_profiler_event("recovery_complete", {
            "recovery_time_ms": recovery_time_ms,
            "new_config": new_config,
            "timestamp": datetime.now().isoformat()
        })
        
        # 복구 완료 마커
        if self.f:
            self.f.write(f"# RECOVERY_COMPLETE: {recovery_time_ms:.2f}ms, new_config={json.dumps(new_config)}\n")
            self.f.flush()

    def _log_profiler_event(self, event_type, details=None):
        """프로파일러 이벤트 로깅"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details or {}
        }
        self.failover_events.append(event)
        
        # 콘솔 출력
        Log.i(f"📊 Profiler Event: {event_type} | {json.dumps(details, ensure_ascii=False)}")

    def get_failover_summary(self):
        """Failover 요약 정보 반환"""
        return {
            "restart_count": self.restart_count,
            "failover_mode": self.failover_mode,
            "total_events": len(self.failover_events),
            "events": self.failover_events,
            "last_known_config": self.last_known_good_config
        }

    def reset_failover_state(self):
        """Failover 상태 리셋"""
        self.failover_mode = False
        self.restart_count = 0
        self.failover_events.clear()
        Log.i("🔄 Profiler failover state reset")


class RemoteTSPipeProfiler(TSPipeProfiler):
    def __init__(self, remote_queue):
        self.profiler_message_queue = remote_queue


def profile_semantic(*args):
    if current_profiler_container.profiler is not None:
        current_profiler_container.profiler.profiler_message_queue.put([timestamp(), *args])


def profile_inject(r):
    if current_profiler_container.profiler is not None:
        current_profiler_container.profiler.profiler_message_queue.put(r)


def profile_init():
    print("[DEBUG] profile_init: profiler =", current_profiler_container.profiler)
    print("[DEBUG] profile_init: is_master =", is_master_process)
    print("[DEBUG] profile_init: f is None =", current_profiler_container.profiler.f is None)

    if current_profiler_container.profiler is not None:
        if is_master_process and current_profiler_container.profiler.f is None:
            self = current_profiler_container.profiler
            self.f = open(self.filename, 'w')

            def log_saver():
                Log.v("Waiting for logs...")
                while self.running:
                    # i = 0
                    while not self.profiler_message_queue.empty():
                        a = self.profiler_message_queue.get()
                        ts, args, kwargs = a[0], a[1:], {}
                        s = self.profile_semantic(ts, *args, **kwargs)
                        print(f"[DEBUG] Profiler saving: {s}")
                        self.f.write(s+"\n")
                    self.f.flush()
                    sleep(0.25)
            self.thd = Thread(target=log_saver, args=())
            self.thd.name = 'ProfilerThread'
            self.thd.start()
            Log.v("Logger Started.")


def remote_profile_init(remote_queue):
    current_profiler_container.profiler = RemoteTSPipeProfiler(remote_queue)


class ProfilerDelegateWorker:
    def __init__(self, queue: 'DistributedQueue'):
        self.queue = queue

        def _thread_main():
            log_out_queue = queue
            while True:
                r = log_out_queue.get()
                # print(f"[DEBUG] ProfilerDelegateWorker received: {r}")
                
                if r is None:
                    print("Terminating Log Delegate Queue")
                    break
                profile_inject(r)

        self.thread = Thread(target=_thread_main)
        self.thread.name = 'ThreadLogDelegate'
        self.thread.start()

    def join(self):
        # self.thread.join()
        print("[DEBUG] Sending STOP to ProfilerDelegateWorker")
        self.queue.put(None)
        print("[DEBUG] STOP sent, joining profiler delegate thread")

        self.thread.join(timeout=10)
        if self.thread.is_alive():
            print("[DEBUG] ProfilerDelegateWorker thread did NOT join gracefully")
        else:
            print("[DEBUG] ProfilerDelegateWorker thread joined successfully")


class Operation:
    def __init__(self,
                 op_name: str,
                 batch_idx: Optional[int] = None,
                 view_idx: Optional[int] = None,
                 ubatch_idx: Optional[int] = None,
                 is_target: Optional[bool] = None,
                 src_partition: Optional[int] = None,
                 dst_partition: Optional[int] = None):
        self.op_name = op_name
        self.batch_idx = batch_idx
        self.view_idx = view_idx
        self.ubatch_idx = ubatch_idx
        self.is_target = is_target
        self.src_partition = src_partition
        self.dst_partition = dst_partition
        self.start_ts = None
        self.end_ts = None

    def __enter__(self):
        self.start_ts = timestamp()
        if current_profiler_container.profiler is not None:
            current_profiler_container.profiler.profiler_message_queue.put([
                self.start_ts, self.batch_idx, self.view_idx, self.ubatch_idx, self.is_target,
                self.src_partition, self.dst_partition, self.op_name,
            ])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_ts = timestamp()
        if current_profiler_container.profiler is not None:
            current_profiler_container.profiler.profiler_message_queue.put([
                self.end_ts, self.batch_idx, self.view_idx, self.ubatch_idx, self.is_target,
                self.src_partition, self.dst_partition, self.op_name+"_finish",
            ])
