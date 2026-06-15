"""
Progress Tracker
훈련 진행 상황 추적 및 K_rem (남은 step 수) 계산

K_rem = K_total - K_done 공식 기반
"""
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class TrainingProgress:
    """훈련 진행 상황 정보"""
    total_epochs: int = 1
    current_epoch: int = 0
    steps_per_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_completion_time: Optional[float] = None
    
    @property
    def remaining_steps(self) -> int:
        """남은 step 수 (K_rem) 계산"""
        if self.total_steps > 0:
            return max(0, self.total_steps - self.current_step)
        return 0
    
    @property
    def progress_ratio(self) -> float:
        """전체 훈련에서 완료된 비율 (0.0 ~ 1.0)"""
        if self.total_steps > 0:
            return min(1.0, self.current_step / self.total_steps)
        return 0.0
    
    @property
    def elapsed_time(self) -> float:
        """훈련 시작부터 경과 시간 (초)"""
        return time.time() - self.start_time

class ProgressTracker:
    """훈련 진행 상황 추적 및 K_rem 계산"""
    
    def __init__(self, total_epochs: int = 1, steps_per_epoch: int = 0):
        self.logger = logging.getLogger(f"{__name__}.ProgressTracker")
        self.progress = TrainingProgress(
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_epochs * steps_per_epoch if steps_per_epoch > 0 else 0
        )
        self._step_times = []  # 최근 step들의 실행 시간 기록
        self._last_step_time = time.time()
        
        self.logger.info(f"📈 ProgressTracker initialized: {total_epochs} epochs x {steps_per_epoch} steps = {self.progress.total_steps} total steps")
    
    def update_step(self, step_id: int, epoch: int = 0) -> int:
        """
        훈련 step 완료 시 진행상황 업데이트
        
        Args:
            step_id: 완료된 step 번호
            epoch: 현재 epoch 번호
            
        Returns:
            int: 남은 step 수 (K_rem)
        """
        current_time = time.time()
        step_duration = current_time - self._last_step_time
        
        # progress 업데이트
        self.progress.current_step = step_id
        self.progress.current_epoch = epoch
        
        # step 시간 기록 (최근 10개만 유지)
        self._step_times.append(step_duration)
        if len(self._step_times) > 10:
            self._step_times.pop(0)
            
        self._last_step_time = current_time
        
        # 완료 시간 예측 업데이트
        self._update_completion_estimate()
        
        K_rem = self.progress.remaining_steps
        
        if step_id % 100 == 0:  # 100 step마다 로깅
            progress_pct = self.progress.progress_ratio * 100
            elapsed_min = self.progress.elapsed_time / 60
            self.logger.info(f"📊 Progress: {progress_pct:.1f}% ({step_id}/{self.progress.total_steps}), K_rem={K_rem}, Elapsed={elapsed_min:.1f}min")
            
        return K_rem
    
    def get_remaining_steps(self) -> int:
        """현재 남은 step 수 반환 (K_rem)"""
        return self.progress.remaining_steps
    
    def get_progress_info(self) -> Dict[str, Any]:
        """상세 진행 정보 반환"""
        avg_step_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0
        
        return {
            "current_step": self.progress.current_step,
            "total_steps": self.progress.total_steps, 
            "remaining_steps": self.progress.remaining_steps,
            "progress_ratio": self.progress.progress_ratio,
            "elapsed_time_sec": self.progress.elapsed_time,
            "avg_step_time_sec": avg_step_time,
            "estimated_completion_time": self.progress.estimated_completion_time,
            "estimated_remaining_time": self._estimate_remaining_time()
        }
    
    def _update_completion_estimate(self):
        """완료 시간 예측 업데이트"""
        if len(self._step_times) < 3:  # 충분한 데이터가 없으면 스키핑
            return
            
        avg_step_time = sum(self._step_times) / len(self._step_times)
        remaining_time = self.progress.remaining_steps * avg_step_time
        
        self.progress.estimated_completion_time = time.time() + remaining_time
    
    def _estimate_remaining_time(self) -> Optional[float]:
        """남은 훈련 시간 예측 (초)"""
        if len(self._step_times) < 3:
            return None
            
        avg_step_time = sum(self._step_times) / len(self._step_times)
        return self.progress.remaining_steps * avg_step_time
    
    def is_late_stage(self, threshold: float = 0.9) -> bool:
        """
        훈련이 후반부인지 확인 (KEEP vs REPLAN 결정에 중요)
        
        Args:
            threshold: 후반부로 간주할 진행률 기준 (기본값: 90%)
            
        Returns:
            bool: True if 후반부, False if 초/중반부
        """
        return self.progress.progress_ratio >= threshold
    
    def reset(self, total_epochs: int = None, steps_per_epoch: int = None):
        """진행상황 리셋 (새로운 훈련 시작 시)"""
        if total_epochs is not None:
            self.progress.total_epochs = total_epochs
        if steps_per_epoch is not None:
            self.progress.steps_per_epoch = steps_per_epoch
            
        self.progress.total_steps = self.progress.total_epochs * self.progress.steps_per_epoch
        self.progress.current_step = 0
        self.progress.current_epoch = 0
        self.progress.start_time = time.time()
        self.progress.estimated_completion_time = None
        
        self._step_times.clear()
        self._last_step_time = time.time()
        
        self.logger.info(f"🔄 ProgressTracker reset: {self.progress.total_steps} total steps")

class FailoverDecisionContext:
    """Failover 결정에 필요한 종합 정보"""
    
    def __init__(self, progress_tracker: ProgressTracker):
        self.progress_tracker = progress_tracker
        self.logger = logging.getLogger(f"{__name__}.FailoverDecisionContext")
    
    def should_consider_failover(self, slowdown_threshold: float = 1.1) -> bool:
        """
        Failover를 고려해야 하는 상황인지 판단
        
        Args:
            slowdown_threshold: Failover 고려 시작점
            
        Returns:
            bool: True if failover 고려해야 함
        """
        # 기본 조건들 체크
        K_rem = self.progress_tracker.get_remaining_steps()
        
        # 1. 남은 step이 너무 적으면 failover 하지 않음
        if K_rem < 10:
            self.logger.debug(f"🚫 Failover skipped: too few remaining steps ({K_rem})")
            return False
        
        # 2. 훈련 시작 직후는 제외 (불안정한 측정값)
        if self.progress_tracker.progress.current_step < 5:
            self.logger.debug(f"🚫 Failover skipped: too early in training")
            return False
            
        return True
    
    def get_context_info(self) -> Dict[str, Any]:
        """의사결정에 필요한 컨텍스트 정보 수집"""
        progress_info = self.progress_tracker.get_progress_info()
        
        return {
            **progress_info,
            "is_late_stage": self.progress_tracker.is_late_stage(),
            "is_very_late_stage": self.progress_tracker.is_late_stage(0.95),  # 95% 완료
            "should_consider_failover": self.should_consider_failover()
        }