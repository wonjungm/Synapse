"""
Alpha-Beta Loader: Runtime lookup 및 분해

학습 중 slowdown이 감지되면,
이 모듈에서 α_comp, β_comm을 빠르게 조회하거나 계산

사용:
  loader = AlphaBetaLoader()
  
  slowdown = 1.15
  alpha_comp, beta_comm = loader.get_alpha_beta(slowdown, num_gpus=4)
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging


class AlphaBetaLoader:
    """Slowdown으로부터 α_comp, β_comm 조회 또는 계산"""
    
    def __init__(self, 
                 table_path: str = "./benchmarks/soft_target/planner/alpha_beta_table.json",
                 enable_fallback: bool = True):
        """
        Args:
            table_path: 프리파일된 alpha_beta 테이블 경로
            enable_fallback: 테이블 없으면 휴리스틱 사용 여부
        """
        self.logger = logging.getLogger(f"{__name__}.AlphaBetaLoader")
        
        self.table_path = Path(table_path)
        self.table: Dict = {}
        self.enable_fallback = enable_fallback
        
        # 테이블 로드 시도
        self._load_table()
    
    def _load_table(self):
        """알파-베타 테이블 로드"""
        
        if not self.table_path.exists():
            self.logger.warning(
                f"⚠️ Alpha-beta table not found: {self.table_path}\n"
                f"   Using fallback heuristic (less accurate)\n"
                f"   Run alpha_beta_profiler.py to create table"
            )
            return
        
        try:
            with open(self.table_path, 'r') as f:
                self.table = json.load(f)
            
            self.logger.info(
                f"✅ Loaded alpha-beta table: {self.table_path}\n"
                f"   Entries: {len(self.table)}"
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to load alpha-beta table: {e}")
    
    def get_alpha_beta(self, 
                      slowdown_ratio: float,
                      num_gpus: int = 4,
                      degraded_gpu_id: int = 0) -> Tuple[float, float]:
        """
        Slowdown ratio로부터 α_comp, β_comm 조회
        
        Args:
            slowdown_ratio: 1.0 = 정상, 1.2 = 20% 느려짐
            num_gpus: 현재 사용 중인 GPU 개수 (K=4 or K=3 등)
            degraded_gpu_id: 느려진 GPU ID (기본: 0)
        
        Returns:
            (alpha_comp, beta_comm)
        """
        
        # 정상 상태
        if slowdown_ratio < 1.02:  # 2% 미만은 정상
            return 1.0, 1.0
        
        # 테이블에서 조회
        if self.table:
            result = self._lookup_table(slowdown_ratio, num_gpus)
            if result is not None:
                return result
        
        # 테이블 미스 또는 테이블 없음 → Fallback
        if self.enable_fallback:
            return self._fallback_heuristic(slowdown_ratio)
        
        return 1.0, 1.0
    
    def _lookup_table(self, 
                     slowdown_ratio: float,
                     num_gpus: int) -> Optional[Tuple[float, float]]:
        """
        테이블에서 가장 가까운 entry 찾기
        
        Returns:
            (alpha_comp, beta_comm) 또는 None
        """
        
        # Scenario key 생성
        scenario_key = f"K{num_gpus}_slowdown_{slowdown_ratio:.2f}"
        
        # 정확한 match 찾기
        if scenario_key in self.table:
            entry = self.table[scenario_key]
            return entry['alpha_comp'], entry['beta_comm']
        
        # 가장 가까운 slowdown 찾기
        best_match = None
        best_distance = float('inf')
        
        for key in self.table.keys():
            if not key.startswith(f"K{num_gpus}_slowdown_"):
                continue
            
            try:
                key_slowdown = float(key.split('_')[-1])
                distance = abs(key_slowdown - slowdown_ratio)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = key
            except:
                continue
        
        if best_match:
            entry = self.table[best_match]
            self.logger.debug(
                f"Found best match for K{num_gpus}, slowdown {slowdown_ratio:.2f}: "
                f"{best_match}"
            )
            return entry['alpha_comp'], entry['beta_comm']
        
        return None
    
    def _fallback_heuristic(self, slowdown_ratio: float) -> Tuple[float, float]:
        """
        Fallback: Compute-dominant 가정
        
        가정: 네트워크는 정상, compute만 느려짐
        β_comm = 1.0 (고정)
        α_comp = slowdown_ratio (직접 사용)
        
        Overhead: <1ms
        
        실제로는:
          slowdown = (α × t_comp + β × t_comm) / (t_comp + t_comm)
          
          가정: β = 1.0, t_comp:t_comm ≈ 3:1 비율
          slowdown = (α × 3 + 1 × 1) / 4
          α = (slowdown × 4 - 1) / 3
          
          근데 간단히 α ≈ slowdown으로 근사해도 대부분 OK
        """
        
        alpha_comp = slowdown_ratio
        beta_comm = 1.0
        
        self.logger.debug(
            f"Using fallback heuristic: α_comp={alpha_comp:.3f}, β_comm={beta_comm:.3f}"
        )
        
        return alpha_comp, beta_comm
    
    def estimate_alpha_beta_inline(self,
                                   slowdown_ratio: float,
                                   t_comp_ms: float,
                                   t_comm_ms: float) -> Tuple[float, float]:
        """
        더 정확한 역산 (Profile 데이터 있을 때)
        
        Args:
            slowdown_ratio: 측정된 slowdown
            t_comp_ms: profile의 compute time
            t_comm_ms: profile의 comm time
        
        Returns:
            (alpha_comp, beta_comm)
        
        공식:
          slowdown = (α × t_comp + β × t_comm) / (t_comp + t_comm)
          
          가정: β = 1.0
          slowdown × (t_comp + t_comm) = α × t_comp + t_comm
          α = (slowdown × (t_comp + t_comm) - t_comm) / t_comp
        """
        
        if t_comp_ms <= 0:
            return slowdown_ratio, 1.0
        
        denominator = slowdown_ratio * (t_comp_ms + t_comm_ms) - t_comm_ms
        alpha_comp = denominator / t_comp_ms
        alpha_comp = max(1.0, alpha_comp)  # 1.0 이상만 의미 있음
        
        beta_comm = 1.0
        
        self.logger.debug(
            f"Estimated α_comp={alpha_comp:.3f}, β_comm={beta_comm:.3f} "
            f"(from slowdown={slowdown_ratio:.3f}, t_comp={t_comp_ms:.1f}ms, "
            f"t_comm={t_comm_ms:.1f}ms)"
        )
        
        return alpha_comp, beta_comm
