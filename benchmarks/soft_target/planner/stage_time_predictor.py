"""
Stage Time Predictor
각 failover 정책(KEEP, REPLAN, DEGRADE)에서의 예상 stage time을 계산

공식: T(p) = max StageTime(g, Π_p(g))  
StageTime(g,l,r) = α_comp(g) * Σt_comp(i) + β_comm(g) * t_comm_out(r)
"""
import json
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .eta_calculator import StageTimeInfo

logger = logging.getLogger(__name__)

@dataclass
class GPUPerformanceState:
    """GPU 성능 상태 정보"""
    gpu_id: int
    current_slowdown: float = 1.0    # 현재 slowdown 비율 (1.0=정상, >1.0=느림)
    alpha_comp: float = 1.0          # compute slowdown 계수
    beta_comm: float = 1.0           # communication slowdown 계수
    is_failed: bool = False          # GPU 장애 여부

@dataclass
class PartitionConfig:
    """파이프라인 분할 구성"""
    snet_partition: List[int]        # SNet 각 stage별 레이어 수
    tnet_partition: List[int]        # TNet 각 stage별 레이어 수
    gpu_assignment: List[int]        # 각 stage별 할당된 GPU ID

class StageTimePredictor:
    """각 정책에서의 stage time 예측"""
    
    def __init__(self, 
                 snet_profile_path: str = './benchmarks/soft_target/planner/profile/snet.csv',
                 tnet_profile_path: str = './benchmarks/soft_target/planner/profile/tnet.csv',
                 bandwidth_gbps: float = 8.0,
                 alpha_beta_path: Optional[str] = None,
                 suppress_alpha_beta_log: bool = False):
        self.logger = logging.getLogger(f"{__name__}.StageTimePredictor")
        
        # Load profiling data
        self.snet_df = pd.read_csv(snet_profile_path)
        self.tnet_df = pd.read_csv(tnet_profile_path)
        
        self.snet_num_layers = len(self.snet_df)
        self.tnet_num_layers = len(self.tnet_df)
        
        # Extract timing and size data
        self._load_profiling_data()
        
        # Network bandwidth
        self.bandwidth_kbps = bandwidth_gbps * 1024 * 1024  # GB/s → KB/s
        
        self.logger.info(f"📊 StageTimePredictor initialized: SNet={self.snet_num_layers} layers, TNet={self.tnet_num_layers} layers")
        # Load alpha/beta from JSON if available, otherwise generate and save
        self.alpha_beta_path = alpha_beta_path
        self.alpha_g = {}
        self.beta_g = {}
        try:
            self._load_or_generate_alpha_beta(self.alpha_beta_path, suppress=suppress_alpha_beta_log)
        except Exception as e:
            self.logger.warning(f"Alpha/beta load/generate failed: {e}")
    
    def _load_profiling_data(self):
        """프로파일링 데이터 로드"""
        # TNet data
        self.tnet_forward_times_ms = self.tnet_df['forward_time_ms'].tolist()
        self.tnet_param_sizes_kb = self.tnet_df['parameter_size_kb'].tolist()
        self.tnet_input_activation_sizes_kb = self.tnet_df['input_activation_size_kb'].tolist()
        
        # SNet data  
        self.snet_forward_times_ms = self.snet_df['forward_time_ms'].tolist()
        self.snet_backward_times_ms = self.snet_df['backward_time_ms'].tolist()
        self.snet_param_sizes_kb = self.snet_df['parameter_size_kb'].tolist()
        self.snet_input_activation_sizes_kb = self.snet_df['input_activation_size_kb'].tolist()
        self.snet_output_activation_sizes_kb = self.snet_df['output_activation_size_kb'].tolist()
        
    def predict_stage_times(self, 
                           gpu_states: List[GPUPerformanceState],
                           current_partition: PartitionConfig,
                           alpha_g: Optional[Dict[int, float]] = None,
                           beta_g: Optional[Dict[int, float]] = None) -> StageTimeInfo:
        """
        현재 상황에서 각 정책의 예상 stage time 계산
        
        Args:
            gpu_states: 각 GPU의 현재 성능 상태
            current_partition: 현재 파이프라인 분할 구성
            alpha_g, beta_g: GPU별 성능 계수
            
        Returns:
            StageTimeInfo: 각 정책별 예상 stage time
        """
        # Use provided alpha/beta or generated defaults
        alpha_g = alpha_g if alpha_g is not None else self.alpha_g
        beta_g = beta_g if beta_g is not None else self.beta_g

        # 1. KEEP: 현재 분할 + 느려진 GPU 성능 그대로 사용
        T_keep = self._calculate_keep_stage_time(gpu_states, current_partition, alpha_g, beta_g)
        
        # 2. REPLAN: 현재 사용 가능한 모든 GPU로 최적 재분할
        available_gpus = [gpu.gpu_id for gpu in gpu_states if not gpu.is_failed]
        T_replan = self._calculate_replan_stage_time(available_gpus, alpha_g, beta_g)
        
        # 3. DEGRADE: 장애 GPU + 가장 느린 GPU 하나 제외하고 재분할
        # ✅ Fix #2 (Step 2): ETA 계산이 실행과 동일한 GPU 집합(가장 느린 GPU 하나만 제외)을 사용
        failed_gpus = [gpu.gpu_id for gpu in gpu_states if gpu.is_failed]
        
        # 가장 느린 GPU를 명시적으로 찾음 (identify_slow_gpu()와 동일 기준)
        affected_slow_gpu = None
        max_slowdown = 1.0
        for gpu in gpu_states:
            if gpu.current_slowdown > max_slowdown:
                max_slowdown = gpu.current_slowdown
                affected_slow_gpu = gpu.gpu_id
        
        # DEGRADE는 failed GPU와 affected slow GPU(가장 느린 것)를 제외한 GPU들로 계산
        degrade_gpus = [
            gpu.gpu_id
            for gpu in gpu_states
            if not gpu.is_failed and gpu.gpu_id != affected_slow_gpu
        ]
        T_degrade = self._calculate_degrade_stage_time(degrade_gpus, alpha_g, beta_g)
        
        self.logger.info(f"⏱️  Stage Time Prediction: KEEP={T_keep:.3f}s, REPLAN={T_replan:.3f}s, DEGRADE={T_degrade:.3f}s")
        
        return StageTimeInfo(T_keep=T_keep, T_replan=T_replan, T_degrade=T_degrade)

    def _load_or_generate_alpha_beta(self, path: Optional[str] = None, suppress: bool = False):
        """
        Initialize ratio-only alpha/beta defaults.

        Runtime updates should come from measured timing ratios via the optimizer,
        not from static hardware decomposition.
        """

        num_gpus = 4  # 기본값: 4 GPU
        self.alpha_g = {i: 1.0 for i in range(num_gpus)}
        self.beta_g = {i: 1.0 for i in range(num_gpus)}

        if not suppress:
            self.logger.info(
                "✅ Alpha-Beta initialized with ratio-only defaults "
                "(alpha=1.0, beta=1.0)"
            )
    
    def _calculate_keep_stage_time(self, 
                                  gpu_states: List[GPUPerformanceState],
                                  partition: PartitionConfig,
                                  alpha_g: Dict[int, float],
                                  beta_g: Dict[int, float]) -> float:
        """KEEP 정책: 현재 분할 유지, α/β가 이미 slowdown ratio를 인코딩하므로 추가 곱셈 없음."""
        stage_times = []
        
        snet_start = 0
        tnet_start = 0
        
        for stage_idx, gpu_id in enumerate(partition.gpu_assignment):
            gpu_state = next((g for g in gpu_states if g.gpu_id == gpu_id), None)
            if gpu_state is None:
                return float('inf')
            if gpu_state.is_failed:
                # KEEP cannot be valid if current partition includes a failed GPU.
                return float('inf')
                
            snet_layers = partition.snet_partition[stage_idx]
            snet_end = snet_start + snet_layers
            # _get_*_stage_time 내부에서 alpha_g × compute + beta_g × comm 적용됨
            # α/β = current/baseline 이므로 추가 slowdown 곱셈 금지 (double-apply 방지)
            snet_time = self._get_snet_stage_time(snet_start, snet_end - 1, gpu_id, alpha_g, beta_g)
            
            tnet_layers = partition.tnet_partition[stage_idx]
            tnet_end = tnet_start + tnet_layers
            tnet_time = self._get_tnet_stage_time(tnet_start, tnet_end - 1, gpu_id, alpha_g, beta_g)
            
            stage_times.append(max(snet_time, tnet_time))
            
            snet_start = snet_end
            tnet_start = tnet_end

        # 파티션 경계가 실제 layer 수와 일치하는지 검증
        if snet_start != self.snet_num_layers or tnet_start != self.tnet_num_layers:
            self.logger.warning(
                f"KEEP partition boundary mismatch: "
                f"snet {snet_start}/{self.snet_num_layers}, "
                f"tnet {tnet_start}/{self.tnet_num_layers}"
            )
            return float('inf')

        return max(stage_times) if stage_times else float('inf')
    
    def _calculate_replan_stage_time(self, 
                                   available_gpus: List[int],
                                   alpha_g: Dict[int, float], 
                                   beta_g: Dict[int, float]) -> float:
        """REPLAN 정책: 사용 가능한 GPU로 DP 기반 최적 재분할"""
        partition = self.solve_optimal_partition(available_gpus, alpha_g, beta_g)
        if partition is None:
            return float('inf')
        return self.calculate_partition_bottleneck_time(partition, alpha_g, beta_g)
    
    def _calculate_degrade_stage_time(self, 
                                    healthy_gpus: List[int],
                                    alpha_g: Dict[int, float],
                                    beta_g: Dict[int, float]) -> float:
        """DEGRADE 정책: 느린 GPU를 제외한 나머지 GPU로 동일한 runtime 계수 하에 재분할"""
        if not healthy_gpus:
            return float('inf')

        # REPLAN과 동일한 runtime alpha/beta 가정으로 비교하되,
        # 느린 GPU만 제외한 GPU 집합으로만 최적 재분할을 계산한다.
        degrade_alpha = {
            int(gpu_id): float(alpha_g.get(int(gpu_id), 1.0))
            for gpu_id in healthy_gpus
        }
        degrade_beta = {
            int(gpu_id): float(beta_g.get(int(gpu_id), 1.0))
            for gpu_id in healthy_gpus
        }
        return self._calculate_replan_stage_time(healthy_gpus, degrade_alpha, degrade_beta)

    def solve_optimal_partition(
        self,
        gpu_ids: List[int],
        alpha_g: Dict[int, float],
        beta_g: Dict[int, float],
    ) -> Optional[PartitionConfig]:
        """Run minimax contiguous DP and return an executable partition config."""
        if not gpu_ids:
            return None

        ordered_gpus = [int(g) for g in gpu_ids]
        max_stages = min(len(ordered_gpus), self.snet_num_layers, self.tnet_num_layers)
        if max_stages <= 0:
            return None
        ordered_gpus = ordered_gpus[:max_stages]

        snet_partition, _ = self._solve_minimax_dp_partition(
            num_layers=self.snet_num_layers,
            gpu_ids=ordered_gpus,
            stage_time_fn=lambda l, r, gpu_id: self._get_snet_stage_time(l, r, gpu_id, alpha_g, beta_g),
        )
        tnet_partition, _ = self._solve_minimax_dp_partition(
            num_layers=self.tnet_num_layers,
            gpu_ids=ordered_gpus,
            stage_time_fn=lambda l, r, gpu_id: self._get_tnet_stage_time(l, r, gpu_id, alpha_g, beta_g),
        )

        if not snet_partition or not tnet_partition:
            return None

        return PartitionConfig(
            snet_partition=snet_partition,
            tnet_partition=tnet_partition,
            gpu_assignment=ordered_gpus,
        )

    def calculate_partition_bottleneck_time(
        self,
        partition: PartitionConfig,
        alpha_g: Dict[int, float],
        beta_g: Dict[int, float],
    ) -> float:
        """Compute pipeline bottleneck time for a concrete partition."""
        if not partition.gpu_assignment:
            return float("inf")

        snet_start = 0
        tnet_start = 0
        stage_times: List[float] = []

        for stage_idx, gpu_id in enumerate(partition.gpu_assignment):
            if stage_idx >= len(partition.snet_partition) or stage_idx >= len(partition.tnet_partition):
                return float("inf")

            snet_layers = int(partition.snet_partition[stage_idx])
            tnet_layers = int(partition.tnet_partition[stage_idx])
            if snet_layers <= 0 or tnet_layers <= 0:
                return float("inf")

            snet_end = snet_start + snet_layers
            tnet_end = tnet_start + tnet_layers

            snet_time = self._get_snet_stage_time(snet_start, snet_end - 1, int(gpu_id), alpha_g, beta_g)
            tnet_time = self._get_tnet_stage_time(tnet_start, tnet_end - 1, int(gpu_id), alpha_g, beta_g)
            stage_times.append(max(snet_time, tnet_time))

            snet_start = snet_end
            tnet_start = tnet_end

        if snet_start != self.snet_num_layers or tnet_start != self.tnet_num_layers:
            return float("inf")

        return max(stage_times) if stage_times else float("inf")

    def _solve_minimax_dp_partition(
        self,
        num_layers: int,
        gpu_ids: List[int],
        stage_time_fn,
    ) -> Tuple[List[int], List[float]]:
        # 최대 stage 시간을 최소화하는 DP로 partition 크기와 stage 시간을 계산.
        num_stages = len(gpu_ids)
        if num_layers <= 0 or num_stages <= 0 or num_layers < num_stages:
            return [], []

        dp = np.full((num_layers + 1, num_stages + 1), np.inf, dtype=float)
        split = np.full((num_layers + 1, num_stages + 1), -1, dtype=int)
        dp[0][0] = 0.0

        for i in range(1, num_layers + 1):
            max_stage = min(i, num_stages)
            for s in range(1, max_stage + 1):
                gpu_id = gpu_ids[s - 1]
                for j in range(s - 1, i):
                    stage_t = stage_time_fn(j, i - 1, gpu_id)
                    candidate = max(dp[j][s - 1], stage_t)
                    if candidate < dp[i][s]:
                        dp[i][s] = candidate
                        split[i][s] = j

        if not np.isfinite(dp[num_layers][num_stages]):
            return [], []

        partition_rev: List[int] = []
        stage_times_rev: List[float] = []
        cur = num_layers
        for s in range(num_stages, 0, -1):
            prev = int(split[cur][s])
            if prev < 0:
                return [], []
            gpu_id = gpu_ids[s - 1]
            partition_rev.append(cur - prev)
            stage_times_rev.append(stage_time_fn(prev, cur - 1, gpu_id))
            cur = prev

        partition = list(reversed(partition_rev))
        stage_times = list(reversed(stage_times_rev))
        return partition, stage_times

    def _solve_minimax_dp(self,
                          num_layers: int,
                          gpu_ids: List[int],
                          stage_time_fn) -> List[float]:
        """
        Minimax DP partition solver.

        For fixed stage-to-GPU assignment (gpu_ids order), find contiguous partitions that
        minimize the maximum stage time.
        """
        _, stage_times = self._solve_minimax_dp_partition(num_layers, gpu_ids, stage_time_fn)
        return stage_times
    
    def _get_snet_stage_time(self, layer_start: int, layer_end: int, gpu_id: int,
                           alpha_g: Dict[int, float], beta_g: Dict[int, float]) -> float:
        # SNet stage time 계산 (optimizer.py 로직 사용)
        if layer_start > layer_end or layer_end >= self.snet_num_layers:
            return 0.0
            
        # Forward + Backward time
        fwd_time = sum(self.snet_forward_times_ms[layer_start:layer_end+1]) * 1e-3  # ms -> s
        bwd_time = sum(self.snet_backward_times_ms[layer_start:layer_end+1]) * 1e-3  # ms -> s
        
        # Communication time (activation transfer)
        recv_act_time = 0
        if layer_start != 0:
            recv_act_size = self.snet_input_activation_sizes_kb[layer_start]
            recv_act_time = recv_act_size / self.bandwidth_kbps
            
        # GPU별 alpha_g와 beta_g 적용
        total_time = (alpha_g.get(gpu_id, 1.0) * (fwd_time + bwd_time) + 
                     beta_g.get(gpu_id, 1.0) * recv_act_time)
        
        return total_time
    
    def _get_tnet_stage_time(self, layer_start: int, layer_end: int, gpu_id: int,
                           alpha_g: Dict[int, float], beta_g: Dict[int, float]) -> float:
        # TNet stage time 계산 (optimizer.py 로직 사용)
        if layer_start > layer_end or layer_end >= self.tnet_num_layers:
            return 0.0
            
        # Forward time only
        fwd_time = sum(self.tnet_forward_times_ms[layer_start:layer_end+1]) * 1e-3  # ms -> s
        
        # Communication time  
        recv_act_time = 0
        if layer_start != 0:
            recv_act_size = self.tnet_input_activation_sizes_kb[layer_start]
            recv_act_time = recv_act_size / self.bandwidth_kbps
            
        # GPU별 alpha_g와 beta_g 적용
        total_time = (alpha_g.get(gpu_id, 1.0) * fwd_time + 
                     beta_g.get(gpu_id, 1.0) * recv_act_time)
        
        return total_time

