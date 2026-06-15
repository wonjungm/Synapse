#!/usr/bin/env python3
"""
GPU 사용률 자동 확인 및 할당 유틸리티
"""
import subprocess
import json
from typing import List, Dict, Tuple, Optional


def get_gpu_status() -> List[Dict]:
    """현재 GPU 상태 정보를 반환"""
    try:
        result = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,name,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)
        
        gpu_status = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpu_status.append({
                    'index': int(parts[0]),
                    'utilization': int(parts[1]) if parts[1] != 'N/A' else 0,
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'name': parts[4],
                    'temperature': int(parts[5]) if len(parts) > 5 and parts[5] != 'N/A' else 0,
                    'memory_percent': round((int(parts[2]) / int(parts[3])) * 100, 1)
                })
        
        return gpu_status
        
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi 실행 실패: {e}")
        return []
    except Exception as e:
        print(f"GPU 상태 확인 실패: {e}")
        return []


def find_available_gpus(max_utilization: int = 95, max_memory_percent: float = 90.0) -> List[int]:
    """사용 가능한 GPU 목록을 반환 (점유율 95% 미만이면 여유공간 있다고 판단)"""
    gpu_status = get_gpu_status()
    available_gpus = []
    
    for gpu in gpu_status:
        if (gpu['utilization'] < max_utilization and 
            gpu['memory_percent'] < max_memory_percent):
            available_gpus.append(gpu['index'])
    
    return available_gpus


def suggest_gpu_allocation(min_gpus_needed: int = 3) -> Dict:
    """실험용 GPU 할당 제안 (4개 GPU 필수)"""
    gpu_status = get_gpu_status()
    available_gpus = find_available_gpus()  # 사용률 < 90%
    
    # 매우 여유로운 GPU들 (사용률 < 10%, 메모리 < 20%)
    idle_gpus = [
        gpu['index'] for gpu in gpu_status 
        if gpu['utilization'] < 10 and gpu['memory_percent'] < 20.0
    ]
    
    # 어느정도 여유로운 GPU들 (사용률 < 80%, 메모리 < 80%)
    low_usage_gpus = [
        gpu['index'] for gpu in gpu_status 
        if gpu['utilization'] < 80 and gpu['memory_percent'] < 80.0
    ]
    
    print("🔍 GPU 상태 분석:")
    print("-" * 80)
    print(f"{'GPU':<3} {'사용률':<6} {'메모리':<15} {'온도':<6} {'상태':<10} {'GPU 이름'}")
    print("-" * 80)
    
    for gpu in gpu_status:
        gpu_id = gpu['index']
        if (gpu_id in idle_gpus):
            status = "✅ 사용가능"
        elif (gpu_id in low_usage_gpus):
            status = "⚠️ 여유있음"  
        elif (gpu_id in available_gpus):
            status = "🔶 일부여유"
        else:
            status = "🚫 거의만석"
            
        print(f"{gpu['index']:<3} {gpu['utilization']:<6}% "
              f"{gpu['memory_used']:<6}/{gpu['memory_total']:<6}MB "
              f"{gpu['temperature']:<6}°C {status:<10} {gpu['name']}")
    
    print("-" * 80)
    
    # 할당 제안 (3개GPU 필수 - 검증용)
    allocation = {
        'available_gpus': available_gpus,
        'idle_gpus': idle_gpus,
        'low_usage_gpus': low_usage_gpus,
        'recommended_experiment_gpus': [],
        'recommended_failover_target': None,
        'experiment_feasible': False,
        'experiment_type': 'none',
        'required_gpus': 3  # 검증용으로 3개로 변경
    }
    
    if len(available_gpus) >= 5:
        # 이상적: 5개 이상의 사용가능 GPU (3개 실험용 + 1개 실패대상 + 1개 백업)
        allocation['recommended_experiment_gpus'] = available_gpus[:3]  # 처음 3개 사용
        allocation['recommended_failover_target'] = available_gpus[3]   # 4번째를 실패 대상으로
        allocation['experiment_feasible'] = True
        allocation['experiment_type'] = 'full'
        
    elif len(available_gpus) >= 3:
        # 충분: 3-4개 사용가능 GPU (전부 실험용으로 사용)
        allocation['recommended_experiment_gpus'] = available_gpus[:3]  # 3개 모두 실험용
        allocation['recommended_failover_target'] = available_gpus[2]   # 3번째를 실패 시뮬레이션용으로
        allocation['experiment_feasible'] = True
        allocation['experiment_type'] = 'shared'  # 공유 환경 
        
    else:
        # 불가능: 3개 미만의 GPU
        allocation['experiment_feasible'] = False
        allocation['experiment_type'] = 'insufficient'
    
    return allocation


def print_allocation_summary(allocation: Dict):
    """할당 요약 출력"""
    print()
    print("📋 GPU 할당 제안:")
    print("=" * 50)
    
    if allocation['experiment_type'] == 'insufficient':
        print("❌ 실험 불가능 - GPU 포화상태")
        print(f"   - 필요한 GPU: 3개 이상 (검증용)")
        print(f"   - 여유있는 GPU: {len(allocation['available_gpus'])}개 ({allocation['available_gpus']})")
        print("   - 거의 모든 GPU가 95% 이상 점유중")
        print("   - 잠시 후 다시 시도하거나 새벽시간대 이용 권장")
        
    elif allocation['experiment_type'] == 'minimal':
        print("⚠️ 최소 실험 가능 (검증용)")
        print(f"   - 실험용 GPU: {allocation['recommended_experiment_gpus']} (3개)")
        print(f"   - 실패 대상 GPU: {allocation['recommended_failover_target']}")
        print("   - 다른 사용자와 GPU 공유 상태")
        print("   - 성능 제한적이지만 실험 진행은 가능")
        
    elif allocation['experiment_type'] == 'shared':
        print("🔶 공유 환경에서 실험 가능 (검증용)")
        print(f"   - 실험용 GPU: {allocation['recommended_experiment_gpus']} (3개)")
        print(f"   - 실패 대상 GPU: {allocation['recommended_failover_target']} (마지막 GPU)")
        print("   - 다른 사용자와 일부 공유하지만 3개 GPU로 검증")
        print("   - 기본적인 failover 테스트 가능")
        
    elif allocation['experiment_type'] == 'full':
        print("✅ 완전한 실험 가능 (공유환경)")
        print(f"   - 실험용 GPU: {allocation['recommended_experiment_gpus']} (3개)")
        print(f"   - 실패 대상 GPU: {allocation['recommended_failover_target']}")
        print("   - 다른 사용자와 일부 공유하지만 충분한 여유공간")
        print("   - 모든 failover 기능 테스트 가능")
    
    print("=" * 50)
    print()


def wait_for_available_gpus(required_gpus: int = 2, check_interval: int = 60, max_wait: int = 1800):
    """GPU가 사용 가능해질 때까지 대기"""
    import time
    
    print(f"⏳ {required_gpus}개 이상의 GPU가 사용 가능해질 때까지 대기 중...")
    print(f"   - 체크 간격: {check_interval}초")
    print(f"   - 최대 대기: {max_wait}초 ({max_wait//60}분)")
    print("   - Ctrl+C로 중단 가능")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < max_wait:
            available_gpus = find_available_gpus()
            
            if len(available_gpus) >= required_gpus:
                print(f"✅ {len(available_gpus)}개 GPU 사용 가능해짐: {available_gpus}")
                return True
            
            elapsed = int(time.time() - start_time)
            print(f"   - 대기 중... ({elapsed}초 경과, 사용가능 GPU: {len(available_gpus)}개)")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n⏹️ 대기가 사용자에 의해 중단되었습니다.")
        return False
    
    print(f"⏰ 최대 대기 시간({max_wait}초) 초과")
    return False


def generate_tspipe_config(experiment_gpus: List[int], config_template: str = "tspipe.yaml") -> str:
    """실험용 GPU에 맞는 TSPipe 설정 생성"""
    import yaml
    import os
    
    try:
        # 기본 설정 로드
        with open(config_template, 'r') as f:
            config = yaml.safe_load(f)
        
        num_gpus = len(experiment_gpus)
        
        if num_gpus == 0:
            # 시뮬레이션만 - 기본 설정 유지하되 매우 작게
            config['tspipe']['model_split']['online'] = [1, 1, 1, 1]
            config['tspipe']['model_split']['target'] = [1, 1, 1, 1]
        elif num_gpus == 1:
            # 1 GPU - 모든 레이어를 한 GPU에
            config['tspipe']['model_split']['online'] = [23]  # 전체 resnet 레이어
            config['tspipe']['model_split']['target'] = [18]  # 전체 vit 레이어
        elif num_gpus == 2:
            # 2 GPU - 반반 분할
            config['tspipe']['model_split']['online'] = [12, 11]
            config['tspipe']['model_split']['target'] = [9, 9]
        elif num_gpus == 3:
            # 3 GPU - 거의 균등 분할
            config['tspipe']['model_split']['online'] = [8, 8, 7]
            config['tspipe']['model_split']['target'] = [6, 6, 6]
        else:
            # 4+ GPU - 기본 4GPU 설정 사용
            config['tspipe']['model_split']['online'] = [6, 6, 6, 5]
            config['tspipe']['model_split']['target'] = [5, 5, 4, 4]
        
        # 동적 설정 파일 이름 생성
        config_filename = f"tspipe_auto_{num_gpus}gpu.yaml"
        
        with open(config_filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📝 자동 생성된 설정 파일: {config_filename}")
        print(f"   - 사용 GPU: {experiment_gpus}")
        print(f"   - Online 분할: {config['tspipe']['model_split']['online']}")
        print(f"   - Target 분할: {config['tspipe']['model_split']['target']}")
        
        return config_filename
        
    except Exception as e:
        print(f"⚠️ 설정 파일 생성 실패: {e}")
        return config_template


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU 자동 할당 도구")
    parser.add_argument("--wait", action="store_true", help="GPU가 사용 가능해질 때까지 대기")
    parser.add_argument("--min-gpus", type=int, default=2, help="최소 필요 GPU 수")
    parser.add_argument("--generate-config", action="store_true", help="TSPipe 설정 파일 생성")
    
    args = parser.parse_args()
    
    allocation = suggest_gpu_allocation(args.min_gpus)
    print_allocation_summary(allocation)
    
    if args.wait and not allocation['experiment_feasible']:
        success = wait_for_available_gpus(args.min_gpus)
        if success:
            allocation = suggest_gpu_allocation(args.min_gpus)
            print_allocation_summary(allocation)
    
    if args.generate_config and allocation['recommended_experiment_gpus']:
        config_file = generate_tspipe_config(allocation['recommended_experiment_gpus'])
        print(f"✅ 설정 파일 생성됨: {config_file}")
    
    # JSON 출력 (다른 스크립트에서 사용하기 위해)
    print("\n" + "="*50)
    print("JSON 출력 (스크립트 연동용):")
    print(json.dumps(allocation, indent=2))