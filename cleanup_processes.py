#!/usr/bin/env python3
"""
TSPipe 프로세스 완전 정리 스크립트
포트 충돌과 좀비 프로세스를 방지하기 위한 시스템 정리
"""

import subprocess
import psutil
import signal
import time
import os
from typing import List, Set

def find_tspipe_processes() -> List[psutil.Process]:
    """TSPipe 관련 프로세스들 찾기"""
    tspipe_processes = []
    current_pid = os.getpid()
    parent_pid = os.getppid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            # 현재 정리 스크립트/부모 프로세스는 제외
            if proc.info['pid'] in (current_pid, parent_pid):
                continue

            lowered = cmdline.lower()
            if 'cleanup_processes.py' in lowered:
                continue

            # 실제 잔여 학습/워커 프로세스만 정리 대상으로 포함
            if any(pattern in lowered for pattern in [
                'train_kd_profiling.py',
                'gpu_worker.py',
                'tspipe/communicator.py',
                'benchmarks/soft_target/tspipe/communicator.py'
            ]):
                tspipe_processes.append(proc)
                print(f"📍 Found TSPipe process: PID {proc.info['pid']} - {cmdline[:100]}...")
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return tspipe_processes

def find_processes_using_ports(ports: List[int]) -> Set[int]:
    """특정 포트를 사용하는 프로세스 PID 찾기"""
    pids = set()
    
    for port in ports:
        try:
            result = subprocess.run(
                ['lsof', '-i', f':{port}', '-t'], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        pid = int(line.strip())
                        pids.add(pid)
                        print(f"🔌 Process {pid} using port {port}")
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            continue
    
    return pids

def kill_processes_gracefully(processes: List[psutil.Process], timeout: int = 10):
    """프로세스들을 단계적으로 종료 (SIGTERM -> SIGKILL)"""
    if not processes:
        return
    
    print(f"🛑 Terminating {len(processes)} TSPipe processes...")
    
    # 1단계: 정중한 종료 (SIGTERM)
    for proc in processes:
        try:
            proc.send_signal(signal.SIGTERM)
            print(f"📤 Sent SIGTERM to PID {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
            
    # 대기
    print(f"⏳ Waiting {timeout} seconds for graceful shutdown...")
    time.sleep(timeout)
    
    # 2단계: 강제 종료 (SIGKILL)
    surviving_processes = []
    for proc in processes:
        try:
            if proc.is_running():
                surviving_processes.append(proc)
        except psutil.NoSuchProcess:
            continue
    
    if surviving_processes:
        print(f"💀 Force killing {len(surviving_processes)} remaining processes...")
        for proc in surviving_processes:
            try:
                proc.kill()
                print(f"🔥 Killed PID {proc.pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

def cleanup_pytorch_ipc():
    """PyTorch IPC 파일들 정리"""
    try:
        import tempfile
        import glob
        
        temp_dir = tempfile.gettempdir()
        ipc_patterns = [
            f"{temp_dir}/pytorch_mpi_*",
            f"{temp_dir}/torch-*",
            "/dev/shm/torch_*"
        ]
        
        print("🧹 Cleaning up PyTorch IPC files...")
        for pattern in ipc_patterns:
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    print(f"🗑️ Removed {file_path}")
                except OSError:
                    continue
                    
    except Exception as e:
        print(f"⚠️ IPC cleanup error: {e}")

def main():
    """메인 정리 프로세스"""
    print("🚀 Starting TSPipe process cleanup...")
    
    # 1. TSPipe 프로세스 찾기
    tspipe_procs = find_tspipe_processes()
    
    # 2. 일반적인 PyTorch 분산 포트들 확인
    common_ports = [31101, 29500, 29501, 29502, 12345, 23456]
    port_pids = find_processes_using_ports(common_ports)
    
    # 3. 포트 사용 프로세스를 TSPipe 프로세스 목록에 추가
    all_pids_to_kill = set(proc.pid for proc in tspipe_procs)
    all_pids_to_kill.update(port_pids)
    
    all_procs_to_kill = []
    for pid in all_pids_to_kill:
        try:
            proc = psutil.Process(pid)
            all_procs_to_kill.append(proc)
        except psutil.NoSuchProcess:
            continue
    
    # 4. 프로세스 종료
    if all_procs_to_kill:
        kill_processes_gracefully(all_procs_to_kill)
    else:
        print("✅ No TSPipe processes found to clean up")
    
    # 5. PyTorch IPC 파일 정리  
    cleanup_pytorch_ipc()
    
    # 6. 최종 확인
    remaining = find_tspipe_processes()
    if remaining:
        print(f"⚠️ Warning: {len(remaining)} processes still running")
        for proc in remaining:
            try:
                cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else ''
                print(f"  - PID {proc.pid}: {cmdline[:80]}...")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    else:
        print("✅ All TSPipe processes cleaned up successfully!")
    
    print("\n🎯 Cleanup complete! Ready for fresh TSPipe execution.")

if __name__ == "__main__":
    main()