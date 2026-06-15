"""
ETA (Estimated Time to Arrival) Calculator
수학적 모델 기반 failover 정책 결정을 위한 핵심 모듈

공식: ETA(p) = C_restart(p) + K_rem * T(p)
목표: p* = argmin ETA(p) for p in {KEEP, REPLAN, DEGRADE}
"""
import time
import os
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class Policy(Enum):
    KEEP = "keep"
    REPLAN = "replan" 
    DEGRADE = "degrade"

@dataclass
class RestartCosts:
    """재시작 관련 비용 정보"""
    C_load: float = 0.0         # 체크포인트 로딩 시간 (초)
    D_replan: float = 0.0       # REPLAN 데이터 재분할 시간 (초)  
    D_degrade: float = 0.0      # DEGRADE 데이터 재분할 시간 (초)
    R_replan: float = 0.0       # REPLAN 재구성 계수
    R_degrade: float = 0.0      # DEGRADE 재구성 계수
    T_base: float = 0.0         # 베이스라인 stage time (초)
    T_opt_K: float = 0.0        # K개 GPU 최적화 시간 (초)
    T_opt_K_minus_1: float = 0.0 # K-1개 GPU 최적화 시간 (초)

@dataclass 
class StageTimeInfo:
    """Stage time 예측 정보"""
    T_keep: float = 0.0         # KEEP 정책 시 예상 stage time (초)
    T_replan: float = 0.0       # REPLAN 정책 시 예상 stage time (초) 
    T_degrade: float = 0.0      # DEGRADE 정책 시 예상 stage time (초)

@dataclass
class ETAResult:
    """ETA 계산 결과"""
    optimal_policy: Policy
    eta_values: Dict[Policy, float]
    costs_breakdown: Dict[str, float]
    
class ETACalculator:
    """ETA 계산 및 최적 정책 결정"""
    
    def __init__(self, restart_costs: RestartCosts):
        self.restart_costs = restart_costs
        self.logger = logging.getLogger(f"{__name__}.ETACalculator")
        
    def calculate_eta(self, 
                     K_rem: int, 
                     stage_times: StageTimeInfo,
                     current_slowdown: float = 1.0) -> ETAResult:
        """
        주어진 조건에서 각 정책의 ETA를 계산하여 최적 정책 결정
        
        Args:
            K_rem: 현재 시점 이후 남아있는 훈련 step 수  
            stage_times: 각 정책에서의 예상 stage time
            current_slowdown: 현재 서능 저하 비율 (1.0 = 정상, >1.0 = 느려짐)
            
        Returns:
            ETAResult: 최적 정책과 상세 계산 결과
        """
        # 1. 각 정책별 ETA 계산
        eta_keep = self._calculate_eta_keep(K_rem, stage_times.T_keep)
        eta_replan = self._calculate_eta_replan(K_rem, stage_times.T_replan)  
        eta_degrade = self._calculate_eta_degrade(K_rem, stage_times.T_degrade)
        
        eta_values = {
            Policy.KEEP: eta_keep,
            Policy.REPLAN: eta_replan,
            Policy.DEGRADE: eta_degrade
        }
        
        # 2. 최적 정책 선택 (최소 ETA)
        optimal_policy = min(eta_values.keys(), key=lambda p: eta_values[p])
        
        # 3. 비용 분석
        costs_breakdown = {
            "K_rem": K_rem,
            "current_slowdown": current_slowdown,
            "eta_keep": eta_keep,
            "eta_replan": eta_replan, 
            "eta_degrade": eta_degrade,
            "optimal_policy": optimal_policy.value,
            "time_saved_vs_keep": eta_keep - eta_values[optimal_policy],
            "restart_cost_replan": eta_replan - K_rem * stage_times.T_replan,
            "restart_cost_degrade": eta_degrade - K_rem * stage_times.T_degrade
        }
        
        self.logger.info(f"🧮 ETA Analysis: K_rem={K_rem}, Optimal={optimal_policy.value}")
        self.logger.info(f"   KEEP: {eta_keep:.2f}s, REPLAN: {eta_replan:.2f}s, DEGRADE: {eta_degrade:.2f}s")
        
        return ETAResult(
            optimal_policy=optimal_policy,
            eta_values=eta_values,
            costs_breakdown=costs_breakdown
        )
    
    def _calculate_eta_keep(self, K_rem: int, T_keep: float) -> float:
        """KEEP 정책의 ETA 계산: ETA(keep) = K_rem * T(keep)"""
        return K_rem * T_keep
    
    def _calculate_eta_replan(self, K_rem: int, T_replan: float) -> float:
        """
        REPLAN 정책의 ETA 계산:
        ETA(replan) = C_load + D_replan + R_replan * T_base + T_opt^(K) + K_rem * T(replan)
        """
        restart_cost = (
            self.restart_costs.C_load
            + self.restart_costs.D_replan
            + self.restart_costs.R_replan * self.restart_costs.T_base
            + self.restart_costs.T_opt_K
        )
        return restart_cost + K_rem * T_replan

    def _calculate_eta_degrade(self, K_rem: int, T_degrade: float) -> float:
        """
        DEGRADE 정책의 ETA 계산:
        ETA(degrade) = C_load + D_degrade + R_degrade * T_base + T_opt^(K-1) + K_rem * T(degrade)
        """
        restart_cost = (
            self.restart_costs.C_load
            + self.restart_costs.D_degrade
            + self.restart_costs.R_degrade * self.restart_costs.T_base
            + self.restart_costs.T_opt_K_minus_1
        )
        return restart_cost + K_rem * T_degrade
        
    def update_restart_costs(self, new_costs: RestartCosts):
        """실험 중 측정된 실제 비용으로 업데이트"""
        self.restart_costs = new_costs
        self.logger.info("🔄 Restart costs updated with measured values")

def create_default_restart_costs() -> RestartCosts:
    """기본값 기반 RestartCosts 생성 (논문 실험용 초기값)"""
    c_load = float(os.environ.get("FAILOVER_RESTART_C_LOAD_SEC", "4.37"))
    d_replan = float(os.environ.get("FAILOVER_RESTART_D_REPLAN_SEC", "14.0"))
    d_degrade = float(os.environ.get("FAILOVER_RESTART_D_DEGRADE_SEC", "10.0"))
    r_replan = float(os.environ.get("FAILOVER_RESTART_R_REPLAN", "50.0"))
    r_degrade = float(os.environ.get("FAILOVER_RESTART_R_DEGRADE", "50.0"))

    return RestartCosts(
        # Conservative default tuned from recent logs:
        # observed restart downtime was ~28-30s around REPLAN.
        # Keep 50*T_base term for scale adaptation and add fixed repartition constants.
        C_load=c_load,
        D_replan=d_replan,
        D_degrade=d_degrade,
        R_replan=r_replan,
        R_degrade=r_degrade,
        T_base=1.0,
        T_opt_K=0.0,
        T_opt_K_minus_1=0.0,
    )