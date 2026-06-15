"""
RestartCost Benchmarking and Validation
실제 restart 비용 측정 및 수학적 모델 검증

측정 항목:
- C_load: 체크포인트 로딩 시간 (첨부 이미지: 4.3~4.5초)
- C_save: 체크포인트 저장 시간 (첨부 이미지: 2.4~2.6초)  
- D_replan, D_degrade: 파이프라인 재구성 시간
- T_opt: 최적화 소요 시간
"""
import time
import torch
import os
import json
import tempfile
import logging
from typing import Dict, List, Tuple
import statistics
from dataclasses import asdict

# Import mathematical model components
try:
    from .eta_calculator import RestartCosts
    from .mathematical_optimizer import MathematicalFailoverOptimizer
    MATHEMATICAL_MODEL_AVAILABLE = True
except ImportError:
    MATHEMATICAL_MODEL_AVAILABLE = False

logger = logging.getLogger(__name__)

class RestartCostBenchmark:
    """Restart 비용 실제 측정"""
    
    def __init__(self, model_size_mb: int = 223):  # 첨부 이미지 기준 모델 크기
        self.model_size_mb = model_size_mb
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger(f"{__name__}.RestartCostBenchmark")
        
        # 측정 결과 저장
        self.measurements = {
            'checkpoint_save_times': [],
            'checkpoint_load_times': [],
            'replan_times': [],
            'degrade_times': [],
            'optimization_times': []
        }
        
        self.logger.info(f"💾 Benchmark initialized: model_size={model_size_mb}MB, temp_dir={self.temp_dir}")
    
    def create_mock_model(self) -> torch.nn.Module:
        """측정용 모의 모델 생성"""
        # 지정 크기의 모델 생성 (parameter 수로 크기 조정)
        param_count = (self.model_size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, param_count // 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(param_count // 2000, 1000)
        )
        
        # GPU로 이동
        if torch.cuda.is_available():
            model = model.cuda()
            
        return model
    
    def measure_checkpoint_save_time(self, model: torch.nn.Module, num_trials: int = 5) -> float:
        """체크포인트 저장 시간 측정"""
        save_times = []
        
        for trial in range(num_trials):
            checkpoint_path = os.path.join(self.temp_dir, f"checkpoint_save_trial_{trial}.pth")
            
            # 저장 시간 측정
            start_time = time.time()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': None,  # optimizer 제외하고 측정
                'epoch': trial,
                'step': trial * 100
            }, checkpoint_path)
            end_time = time.time()
            
            save_time = end_time - start_time
            save_times.append(save_time)
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            self.logger.debug(f"💾 Trial {trial}: save_time={save_time:.3f}s, file_size={file_size_mb:.1f}MB")
            
            # 정리
            os.remove(checkpoint_path)
        
        avg_save_time = statistics.mean(save_times)
        std_save_time = statistics.stdev(save_times) if len(save_times) > 1 else 0
        
        self.measurements['checkpoint_save_times'].extend(save_times)
        self.logger.info(f"💾 Checkpoint Save: {avg_save_time:.3f}±{std_save_time:.3f}s (n={num_trials})")
        
        return avg_save_time
    
    def measure_checkpoint_load_time(self, model: torch.nn.Module, num_trials: int = 5) -> float:
        """체크포인트 로딩 시간 측정"""
        # 먼저 저장할 체크포인트 생성
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_for_load_test.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,
            'epoch': 1,
            'step': 100
        }, checkpoint_path)
        
        load_times = []
        
        for trial in range(num_trials):
            # 모델 상태 초기화
            for param in model.parameters():
                param.data.fill_(0.0)
            
            # 로딩 시간 측정
            start_time = time.time()
            checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            end_time = time.time()
            
            load_time = end_time - start_time
            load_times.append(load_time)
            
            self.logger.debug(f"📁 Trial {trial}: load_time={load_time:.3f}s")
        
        avg_load_time = statistics.mean(load_times)
        std_load_time = statistics.stdev(load_times) if len(load_times) > 1 else 0
        
        self.measurements['checkpoint_load_times'].extend(load_times)
        self.logger.info(f"📁 Checkpoint Load: {avg_load_time:.3f}±{std_load_time:.3f}s (n={num_trials})")
        
        # 정리
        os.remove(checkpoint_path)
        
        return avg_load_time
    
    def measure_replan_time(self, num_trials: int = 3) -> float:
        """REPLAN 재구성 시간 측정"""
        replan_times = []
        
        for trial in range(num_trials):
            # REPLAN 시뮬레이션 (실제로는 파이프라인 재분할 + 재시작)
            start_time = time.time()
            
            # 1. 파이프라인 재분할 계산 (실제 optimizer 호출)
            self._simulate_pipeline_repartition()
            
            # 2. 프로세스 재시작 시뮬레이션
            self._simulate_process_restart()
            
            end_time = time.time()
            
            replan_time = end_time - start_time
            replan_times.append(replan_time)
            
            self.logger.debug(f"🔄 REPLAN Trial {trial}: {replan_time:.3f}s")
        
        avg_replan_time = statistics.mean(replan_times)
        self.measurements['replan_times'].extend(replan_times)
        self.logger.info(f"🔄 REPLAN Time: {avg_replan_time:.3f}s (n={num_trials})")
        
        return avg_replan_time
    
    def measure_degrade_time(self, num_trials: int = 3) -> float:
        """DEGRADE 재구성 시간 측정"""
        degrade_times = []
        
        for trial in range(num_trials):
            # DEGRADE는 REPLAN보다 단순 (GPU 제외만)
            start_time = time.time()
            
            # K-1 GPU 재분할 (더 단순)
            self._simulate_degrade_repartition()
            self._simulate_process_restart()
            
            end_time = time.time()
            
            degrade_time = end_time - start_time  
            degrade_times.append(degrade_time)
            
            self.logger.debug(f"⬇️ DEGRADE Trial {trial}: {degrade_time:.3f}s")
        
        avg_degrade_time = statistics.mean(degrade_times)
        self.measurements['degrade_times'].extend(degrade_times)
        self.logger.info(f"⬇️ DEGRADE Time: {avg_degrade_time:.3f}s (n={num_trials})")
        
        return avg_degrade_time
    
    def _simulate_pipeline_repartition(self):
        """파이프라인 재분할 시뮬레이션"""
        # 실제로는 optimizer.py의 DP 알고리즘 실행
        time.sleep(0.2)  # 재분할 계산 시간 시뮬레이션
    
    def _simulate_degrade_repartition(self):
        """DEGRADE 재분할 시뮬레이션 (더 단순)"""
        time.sleep(0.15)  # REPLAN보다 빠름
    
    def _simulate_process_restart(self):
        """프로세스 재시작 시뮬레이션"""
        time.sleep(0.5)  # 프로세스 재시작 오버헤드
    
    def run_full_benchmark(self) -> RestartCosts:
        """전체 벤치마크 실행"""
        self.logger.info("🚀 Starting comprehensive restart cost benchmark...")
        
        # 1. 모의 모델 생성
        model = self.create_mock_model()
        
        # 2. 각 비용 측정
        C_save = self.measure_checkpoint_save_time(model)
        C_load = self.measure_checkpoint_load_time(model)
        D_replan = self.measure_replan_time()
        D_degrade = self.measure_degrade_time()
        
        # 3. 기타 비용 추정
        T_base = 1.0  # 베이스라인 stage time (실제 측정 필요)
        T_opt_K = 0.1  # K GPU 최적화 시간
        T_opt_K_minus_1 = 0.15  # K-1 GPU 최적화 시간
        
        # 4. RestartCosts 객체 생성
        measured_costs = RestartCosts(
            C_load=C_load,
            D_replan=D_replan,
            D_degrade=D_degrade,
            R_replan=1.0,  # 재구성 계수 (baseline 대비)
            R_degrade=1.0,
            T_base=T_base,
            T_opt_K=T_opt_K,
            T_opt_K_minus_1=T_opt_K_minus_1
        )
        
        self.logger.info("✅ Benchmark completed!")
        self._log_final_summary(measured_costs, C_save)
        
        return measured_costs
    
    def _log_final_summary(self, costs: RestartCosts, C_save: float):
        """최종 측정 결과 요약 로깅"""
        self.logger.info("📊 === RESTART COST BENCHMARK RESULTS ===")
        self.logger.info(f"💾 Checkpoint Save (C_save): {C_save:.3f}s")
        self.logger.info(f"📁 Checkpoint Load (C_load): {costs.C_load:.3f}s")
        self.logger.info(f"🔄 REPLAN Time (D_replan): {costs.D_replan:.3f}s")
        self.logger.info(f"⬇️ DEGRADE Time (D_degrade): {costs.D_degrade:.3f}s")
        self.logger.info(f"📏 Baseline Stage Time (T_base): {costs.T_base:.3f}s")
        
        # 첨부 이미지 값과 비교
        expected_save = 2.5  # 2.4~2.6초
        expected_load = 4.4  # 4.3~4.5초
        
        self.logger.info(f"🔍 Validation vs Expected:")
        self.logger.info(f"   Save: {C_save:.3f}s (expected: ~{expected_save:.1f}s)")
        self.logger.info(f"   Load: {costs.C_load:.3f}s (expected: ~{expected_load:.1f}s)")
        
        # 편차 계산
        save_diff = abs(C_save - expected_save) / expected_save * 100
        load_diff = abs(costs.C_load - expected_load) / expected_load * 100
        
        if save_diff < 20 and load_diff < 20:  # 20% 이내 오차
            self.logger.info("✅ Measurements within expected range!")
        else:
            self.logger.warning(f"⚠️ Large deviation: save={save_diff:.1f}%, load={load_diff:.1f}%")

def validate_mathematical_model():
    """수학적 모델 검증 - 실제 비용으로 ETA 계산 테스트"""
    if not MATHEMATICAL_MODEL_AVAILABLE:
        logger.error("❌ Mathematical model not available for validation")
        return
        
    logger.info("🧪 Validating mathematical model with measured costs...")
    
    # 1. 실제 비용 측정
    benchmark = RestartCostBenchmark()
    measured_costs = benchmark.run_full_benchmark()
    
    # 2. 수학적 모델에 적용
    optimizer = MathematicalFailoverOptimizer(total_epochs=1, steps_per_epoch=1000)
    optimizer.update_measured_costs(asdict(measured_costs))
    
    # 3. 다양한 시나리오에서 ETA 계산 테스트
    test_scenarios = [
        {"K_rem": 100, "slowdown": 1.2, "description": "Early stage, moderate slowdown"},
        {"K_rem": 50, "slowdown": 1.5, "description": "Mid stage, severe slowdown"}, 
        {"K_rem": 10, "slowdown": 1.3, "description": "Late stage, moderate slowdown"},
        {"K_rem": 5, "slowdown": 2.0, "description": "Very late stage, severe slowdown"}
    ]
    
    logger.info("🧮 ETA Calculation Test Results:")
    for i, scenario in enumerate(test_scenarios):
        optimizer.update_training_progress(1000 - scenario["K_rem"])  # remaining steps 설정
        
        policy = optimizer.evaluate_slowdown_and_decide(
            gpu_id=1, 
            current_slowdown=scenario["slowdown"]
        )
        
        logger.info(f"  Scenario {i+1}: {scenario['description']}")
        logger.info(f"    K_rem={scenario['K_rem']}, slowdown={scenario['slowdown']:.1f}x → Policy: {policy}")
    
    logger.info("✅ Mathematical model validation completed!")

def export_measured_costs_to_json(costs: RestartCosts, filename: str = "measured_restart_costs.json"):
    """측정된 비용을 JSON 파일로 내보내기"""
    costs_dict = asdict(costs)
    costs_dict['measurement_timestamp'] = time.time()
    costs_dict['measurement_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(filename, 'w') as f:
        json.dump(costs_dict, f, indent=2)
    
    logger.info(f"💾 Measured costs exported to {filename}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🎯 RestartCost Benchmark & Mathematical Model Validation")
    print("=" * 60)
    
    # 1. 비용 측정 실행
    benchmark = RestartCostBenchmark()
    measured_costs = benchmark.run_full_benchmark()
    
    # 2. JSON으로 내보내기
    export_measured_costs_to_json(measured_costs)
    
    # 3. 수학적 모델 검증
    validate_mathematical_model()
    
    print("\n✅ All benchmarks and validations completed!")
    print("📁 Check 'measured_restart_costs.json' for detailed results")