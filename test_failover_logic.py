#!/usr/bin/env python3
"""
Failover 로직 직접 테스트 스크립트
실제 TSPipe 없이 failover 핵심 기능 검증
"""
import time
import torch
import sys
import os
import argparse
from datetime import datetime

# TSPipe 경로 추가
sys.path.insert(0, '/home/wisekhy/tspipe/Synapse-private')

from tspipe.failover_logger import init_experiment_logger
from tspipe.gpu_health_monitor import GPUHealthMonitor, GPUFailureEvent

class MockTSPipeFailover:
    """TSPipe Failover 로직만 테스트하는 Mock 클래스"""
    
    def __init__(self, target_fail_gpu=4):
        print("🚀 Mock TSPipe Failover 시스템 초기화")
        
        # 기본 변수들
        self.failover_enabled = True
        self.target_fail_gpu = target_fail_gpu
        self.experiment_logger = init_experiment_logger("mock_failover_test")
        self._shutdown_scheduler = False
        self.artifact_dir = "./results/mock_failover"
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        # Mock GPU workers와 파티션 설정
        self.gpu_workers = [self._create_mock_worker(i) for i in range(7)]
        self.original_partition_config = {
            'online': [2, 2, 2, 2, 2, 2, 0],
            'target': [2, 2, 2, 2, 2, 2, 0]
        }
        
        # GPU Health Monitor 초기화 (실제 콜백 연결)
        self.gpu_health_monitor = GPUHealthMonitor(
            failure_callback=self._on_gpu_failure,
            check_interval=2
        )
        
        print("✅ Mock Failover 시스템 초기화 완료")
        
    def _create_mock_worker(self, gpu_id):
        """Mock GPU worker 생성"""
        class MockWorker:
            def __init__(self, gpu_id):
                self.partition_id = gpu_id
                self.device_id = gpu_id
                
        return MockWorker(gpu_id)
        
    def start_test(self):
        """테스트 시작"""
        print(f"🔬 Mock TSPipe Failover 테스트 시작")
        print(f"🎯 Target GPU: {self.target_fail_gpu}")
        
        # GPU Health Monitor 시작
        self.gpu_health_monitor.start_monitoring()
        
        # 3초 후 GPU 실패 강제 시뮬레이션
        print("⏳ 3초 후 GPU 실패 시뮬레이션 시작...")
        time.sleep(3)
        
        print(f"💣 GPU {self.target_fail_gpu} 실패 강제 주입")
        success = self.gpu_health_monitor.force_gpu_failure(
            self.target_fail_gpu, 
            failure_type="mock_test_failure"
        )
        
        if success:
            print("✅ GPU 실패 시뮬레이션 주입 성공")
        else:
            print("❌ GPU 실패 시뮬레이션 주입 실패")
        
        # 10초 대기 (실제 failover 콜백 확인)
        print("⏳ Failover 콜백 대기 중...")
        time.sleep(10)
        
        print("🛑 테스트 종료")
        self.gpu_health_monitor.stop_monitoring()
        
    def _on_gpu_failure(self, failure_event: GPUFailureEvent):
        """🔥 실제 구현한 GPU 실패 이벤트 핸들러 (TSPipe와 동일)"""
        print(f"💥 GPU {failure_event.gpu_id} failure detected! Type: {failure_event.failure_type}")
        
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
        print(f"🛑 EMERGENCY SHUTDOWN: Terminating all processes due to GPU {failure_event.gpu_id} failure")
        self._emergency_shutdown_and_failover(failure_event.gpu_id)

    def _emergency_shutdown_and_failover(self, failed_gpu_id: int):
        """🎯 사용자 연구 목표에 맞는 긴급 종료 및 failover 프로세스"""
        print("🚨 Starting emergency shutdown and failover process...")
        
        # 1. 현재 학습 루프 강제 중단
        self._force_stop_current_training()
        
        # 2. 실패한 GPU worker 즉시 종료
        self._terminate_failed_workers(failed_gpu_id)
        
        # 3. 체크포인트 저장 (현재 상태 백업)
        self._save_emergency_checkpoint()
        
        # 4. K-1 GPU로 재분할 계산 및 재시작 준비
        self._prepare_k_minus_1_restart(failed_gpu_id)
        
        print("✅ Failover 프로세스 완료!")
        print("📋 연구 목표 달성: 장애 감지 → 강제 종료 → 체크포인트 저장 → K-1 GPU 재시작 준비")
        
    def _force_stop_current_training(self):
        """현재 진행 중인 학습 루프 강제 중단"""
        print("⏹️ Forcing stop of current training loop...")
        self._shutdown_scheduler = True
        print("✅ Training loop stopped")
        
    def _terminate_failed_workers(self, failed_gpu_id: int):
        """실패한 GPU의 worker들 즉시 종료"""
        print(f"💀 Terminating workers on failed GPU {failed_gpu_id}")
        
        failed_workers = [w for w in self.gpu_workers if w.device_id == failed_gpu_id]
        for worker in failed_workers:
            print(f"🔥 Force killing worker {worker.partition_id} on GPU {failed_gpu_id}")
        
        print(f"✅ {len(failed_workers)} workers terminated")
        
    def _save_emergency_checkpoint(self):
        """긴급 체크포인트 저장"""
        print("💾 Saving emergency checkpoint...")
        checkpoint_path = f"{self.artifact_dir}/emergency_checkpoint.pth"
        
        # Mock 체크포인트 데이터
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'failed_gpu': self.target_fail_gpu,
            'model_state': 'mock_model_state_dict'
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"✅ Emergency checkpoint saved to {checkpoint_path}")
        
    def _prepare_k_minus_1_restart(self, failed_gpu_id: int):
        """K-1 GPU 재분할 정보 준비"""
        print(f"📋 Preparing K-1 GPU restart configuration (excluding GPU {failed_gpu_id})")
        
        # 사용 가능한 GPU 목록 (실패한 GPU 제외)
        available_gpus = [i for i in range(torch.cuda.device_count()) if i != failed_gpu_id]
        
        print(f"📊 Available GPUs for restart: {available_gpus}")
        print(f"📊 Original partition config: {self.original_partition_config}")
        
        # 재시작 정보를 파일로 저장
        restart_config = {
            'failed_gpu': failed_gpu_id,
            'available_gpus': available_gpus,
            'original_config': self.original_partition_config,
            'checkpoint_path': f"{self.artifact_dir}/emergency_checkpoint.pth",
            'failover_timestamp': time.time()
        }
        
        import json
        restart_config_path = f"{self.artifact_dir}/restart_config.json" 
        with open(restart_config_path, 'w') as f:
            json.dump(restart_config, f, indent=2)
            
        print(f"✅ Restart configuration saved to {restart_config_path}")
        
        # 결과 요약
        print(f"""
🎯 FAILOVER 완료 요약:
┌─────────────────────────────────────┐
│ 1. ✅ 장애 감지: GPU {failed_gpu_id} 실패 감지     │
│ 2. ✅ 강제 종료: 학습 루프 중단       │
│ 3. ✅ 체크포인트: 모델 상태 저장      │
│ 4. ✅ K-1 재시작: {len(available_gpus)}개 GPU 준비        │
└─────────────────────────────────────┘
""")


def main():
    parser = argparse.ArgumentParser(description="Mock TSPipe Failover Test")
    parser.add_argument("--target-gpu", type=int, default=4, help="Target GPU to fail")
    args = parser.parse_args()
    
    print("🧪 Mock TSPipe Failover Logic Test")
    print("=" * 50)
    
    # Mock failover 테스트 실행
    mock_tspipe = MockTSPipeFailover(target_fail_gpu=args.target_gpu)
    mock_tspipe.start_test()
    
if __name__ == "__main__":
    main()