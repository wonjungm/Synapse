"""
Dynamic Policy Selector
수학적 모델 기반 동적 failover 정책 결정

기존 단순 임계치 로직을 ETA 기반 최적 선택으로 대체
"""
import os
import time
import logging
import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from .eta_calculator import ETACalculator, RestartCosts, Policy, create_default_restart_costs
from .stage_time_predictor import StageTimePredictor, GPUPerformanceState, PartitionConfig
from .progress_tracker import ProgressTracker, FailoverDecisionContext

logger = logging.getLogger(__name__)

@dataclass
class FailoverDecision:
    """Failover 결정 결과"""
    recommended_policy: Policy
    confidence_score: float  # 0.0 ~ 1.0, 결정의 확신도
    eta_analysis: Dict[str, float]
    context_info: Dict[str, any]
    decision_timestamp: float = field(default_factory=time.time)
    reasoning: str = ""

@dataclass
class SlowdownEvent:
    """성능 저하 이벤트 정보"""
    gpu_id: int
    slowdown_ratio: float
    detection_time: float
    sustained_duration: float = 0.0
    
class DynamicPolicySelector:
    """수학적 모델 기반 동적 정책 선택"""
    
    def __init__(self,
                 progress_tracker: ProgressTracker,
                 restart_costs: Optional[RestartCosts] = None,
                 alpha_g: Optional[Dict[int, float]] = None,
                 beta_g: Optional[Dict[int, float]] = None):
        self.logger = logging.getLogger(f"{__name__}.DynamicPolicySelector")
        
        # Core components
        self.progress_tracker = progress_tracker
        self.decision_context = FailoverDecisionContext(progress_tracker)
        self.eta_calculator = ETACalculator(restart_costs or create_default_restart_costs())
        self.stage_time_predictor = StageTimePredictor()
        
        # GPU performance coefficients (normalize JSON string keys to int)
        alpha_src = alpha_g or {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        beta_src = beta_g or {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        self.alpha_g = {int(k): float(v) for k, v in alpha_src.items()}
        self.beta_g = {int(k): float(v) for k, v in beta_src.items()}
        
        # Decision history
        self.decision_history: List[FailoverDecision] = []
        self.slowdown_events: Dict[int, SlowdownEvent] = {}  # gpu_id -> event
        
        # Configuration
        self.min_slowdown_threshold = 1.05   # 5% 이상 느려져야 고려
        # ✅ NEW: Allow environment override for testing (default 30.0s for production)
        threshold_str = os.environ.get("FAILOVER_SLOWDOWN_THRESHOLD_SEC", "").strip()
        self.sustained_slowdown_duration = float(threshold_str) if threshold_str else 30.0
        if threshold_str and float(threshold_str) > 0:
            self.logger.info(f"🧪 Slowdown duration threshold overridden: {self.sustained_slowdown_duration}s")

        # REPLAN guardrail: require minimum ETA gain over modeled restart overhead.
        # This is intentionally conservative to avoid loss-making REPLAN triggers.
        self.replan_min_gain_margin = float(
            os.environ.get("FAILOVER_REPLAN_MIN_GAIN_MARGIN", "1.2")
        )
        self.replan_fallback_restart_sec = float(
            os.environ.get("FAILOVER_REPLAN_FALLBACK_RESTART_SEC", "30.0")
        )
        # Optional weak late-stage bias: disabled by default for simpler interpretation.
        self.enable_late_stage_replan_bias = os.environ.get(
            "FAILOVER_ENABLE_LATE_STAGE_REPLAN_BIAS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.late_stage_replan_bias_threshold = float(
            os.environ.get("FAILOVER_LATE_STAGE_REPLAN_BIAS_THRESHOLD", "0.90")
        )
        self.late_stage_replan_margin_multiplier = float(
            os.environ.get("FAILOVER_LATE_STAGE_REPLAN_MARGIN_MULTIPLIER", "1.0")
        )
        # Keep this off by default to avoid extra confounders during paper deadline runs.
        self.enable_late_stage_force_keep = os.environ.get(
            "FAILOVER_ENABLE_LATE_STAGE_FORCE_KEEP", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.late_stage_force_keep_threshold = float(
            os.environ.get("FAILOVER_LATE_STAGE_FORCE_KEEP_THRESHOLD", "0.95")
        )

        # Timing
        self.last_eta_compute_ms: Optional[float] = None
        
        self.logger.info("🧠 DynamicPolicySelector initialized with mathematical model")
    
    def evaluate_slowdown(
        self,
        gpu_id: int,
        current_slowdown: float,
        current_partition: PartitionConfig,
        failed_gpus: Optional[List[int]] = None,
        trigger_confirmed: bool = False,
    ) -> FailoverDecision:
        """
        GPU 성능 저하 감지 시 최적 정책 결정

        trigger_confirmed=False:
            기존처럼 내부 sustained gate 적용
        trigger_confirmed=True:
            upstream(train_kd.py)에서 wall-clock sustained trigger가 이미 확인된 상태
            -> 여기서는 localization/policy만 수행
        """
        failed_gpus = failed_gpus or []
        current_time = time.time()

        if not trigger_confirmed:
            # 1. 기존 per-GPU sustained logic
            self._update_slowdown_event(gpu_id, current_slowdown, current_time)

            if not self.decision_context.should_consider_failover():
                return self._create_keep_decision("Failover conditions not met")

            if current_slowdown < self.min_slowdown_threshold:
                return self._create_keep_decision(
                    f"Slowdown {current_slowdown:.2f} below threshold"
                )

            event = self.slowdown_events.get(gpu_id)
            if event is not None and event.sustained_duration < self.sustained_slowdown_duration:
                return self._create_keep_decision(
                    f"Slowdown not sustained long enough "
                    f"({event.sustained_duration:.1f}s < {self.sustained_slowdown_duration:.1f}s)"
                )
        else:
            # 1. upstream wall-clock trigger confirmed
            if not self.decision_context.should_consider_failover():
                return self._create_keep_decision("Failover conditions not met")

            self.logger.info(
                f"✅ Upstream wall-clock trigger confirmed. "
                f"Bypassing per-GPU sustained gate for GPU{gpu_id} "
                f"(slowdown={current_slowdown:.2f}x)"
            )

        # 2. GPU 상태 구성
        gpu_states = self._build_gpu_states(current_slowdown, gpu_id, failed_gpus)

        # 3. Stage time 예측
        stage_times = self.stage_time_predictor.predict_stage_times(
            gpu_states, current_partition, self.alpha_g, self.beta_g
        )

        # 4. ETA 계산
        eta_start = time.perf_counter()
        K_rem = self.progress_tracker.get_remaining_steps()
        eta_result = self.eta_calculator.calculate_eta(K_rem, stage_times, current_slowdown)
        eta_end = time.perf_counter()
        self.last_eta_compute_ms = (eta_end - eta_start) * 1000.0
        self.logger.info(f"⏱ ETA computation overhead: {self.last_eta_compute_ms:.3f} ms")

        # 5. Hard failure tie-break
        if failed_gpus:
            eta_replan = eta_result.eta_values.get(Policy.REPLAN, float("inf"))
            eta_degrade = eta_result.eta_values.get(Policy.DEGRADE, float("inf"))
            if eta_degrade <= eta_replan:
                eta_result.optimal_policy = Policy.DEGRADE

        # 6. 결정 분석 및 기록
        decision = self._analyze_and_validate_decision(eta_result, K_rem, current_slowdown)
        self.decision_history.append(decision)
        self._log_decision(decision, gpu_id, current_slowdown, K_rem)
        return decision
    
    def _update_slowdown_event(self, gpu_id: int, slowdown: float, current_time: float):
        """Slowdown 이벤트 추적 업데이트"""
        if gpu_id not in self.slowdown_events:
            self.slowdown_events[gpu_id] = SlowdownEvent(
                gpu_id=gpu_id,
                slowdown_ratio=slowdown,
                detection_time=current_time
            )
        else:
            event = self.slowdown_events[gpu_id]
            event.slowdown_ratio = slowdown
            
            # ✅ FIX #3: Reset detection_time when slowdown falls below threshold
            # Prevents false positives from transient spikes
            _MIN_SLOWDOWN_THRESHOLD = 1.05  # Recovery threshold
            if slowdown < _MIN_SLOWDOWN_THRESHOLD:
                # GPU recovered: reset the detection clock
                event.detection_time = current_time
                event.sustained_duration = 0.0
            else:
                # Still slow: accumulate duration
                event.sustained_duration = current_time - event.detection_time
            
    def _build_gpu_states(self, current_slowdown: float, affected_gpu_id: int, failed_gpus: List[int]) -> List[GPUPerformanceState]:
        """현재 GPU 상태 정보 구성"""
        gpu_states = []

        # Build GPU ids from coefficients plus current event/failure ids.
        known_gpu_ids = set(self.alpha_g.keys()) | set(self.beta_g.keys()) | set(failed_gpus) | {affected_gpu_id}
        for gpu_id in sorted(known_gpu_ids):
            is_failed = gpu_id in failed_gpus
            slowdown = current_slowdown if gpu_id == affected_gpu_id else 1.0
            
            gpu_states.append(GPUPerformanceState(
                gpu_id=gpu_id,
                current_slowdown=slowdown,
                alpha_comp=self.alpha_g.get(gpu_id, 1.0),
                beta_comm=self.beta_g.get(gpu_id, 1.0),
                is_failed=is_failed
            ))
            
        return gpu_states
    
    def _analyze_and_validate_decision(self, eta_result, K_rem: int, current_slowdown: float) -> FailoverDecision:
        """ETA 결과 분석 및 결정 검증"""
        optimal_policy = eta_result.optimal_policy
        eta_values = eta_result.eta_values
        replan_guardrail_applied = False

        # Low-risk guardrail: REPLAN only when ETA gain clearly exceeds restart overhead.
        # This suppresses loss-making REPLAN decisions when the cost model is optimistic.
        if optimal_policy == Policy.REPLAN:
            eta_keep = float(eta_values.get(Policy.KEEP, float("inf")))
            eta_replan = float(eta_values.get(Policy.REPLAN, float("inf")))
            expected_gain = eta_keep - eta_replan

            modeled_restart_cost = self._estimate_replan_restart_cost(eta_result)
            effective_margin = self.replan_min_gain_margin

            if self.enable_late_stage_replan_bias and self.progress_tracker.is_late_stage(self.late_stage_replan_bias_threshold):
                effective_margin *= self.late_stage_replan_margin_multiplier

            required_gain = modeled_restart_cost * effective_margin
            if expected_gain <= required_gain:
                eta_result.optimal_policy = Policy.KEEP
                optimal_policy = Policy.KEEP
                replan_guardrail_applied = True
        
        # 신뢰도 계산
        confidence_score = self._calculate_confidence(eta_result, K_rem, current_slowdown)
        
        # 결정 논리 생성
        reasoning = self._generate_reasoning(eta_result, K_rem, current_slowdown)

        if replan_guardrail_applied:
            # Append guardrail trace when KEEP is selected by REPLAN gain gate.
            eta_keep = float(eta_values.get(Policy.KEEP, float("inf")))
            eta_replan = float(eta_values.get(Policy.REPLAN, float("inf")))
            expected_gain = eta_keep - eta_replan
            modeled_restart_cost = self._estimate_replan_restart_cost(eta_result)
            effective_margin = self.replan_min_gain_margin
            if self.enable_late_stage_replan_bias and self.progress_tracker.is_late_stage(self.late_stage_replan_bias_threshold):
                effective_margin *= self.late_stage_replan_margin_multiplier
            required_gain = modeled_restart_cost * effective_margin
            if expected_gain <= required_gain:
                reasoning += (
                    f" [REPLAN guardrail: gain={expected_gain:.1f}s <= "
                    f"required={required_gain:.1f}s (restart={modeled_restart_cost:.1f}s, "
                    f"margin={effective_margin:.2f}) -> KEEP]"
                )
        
        # 특수 상황 처리
        if self.enable_late_stage_force_keep and self.progress_tracker.is_late_stage(self.late_stage_force_keep_threshold):
            if optimal_policy == Policy.REPLAN:
                reasoning += " [Late-stage: REPLAN cost too high, forcing KEEP]"
                optimal_policy = Policy.KEEP
                confidence_score *= 0.8  # 강제 변경으로 인한 신뢰도 감소
        
        context_info = self.decision_context.get_context_info()
        
        return FailoverDecision(
            recommended_policy=optimal_policy,
            confidence_score=confidence_score,
            eta_analysis=eta_values,
            context_info=context_info,
            reasoning=reasoning
        )

    def _estimate_replan_restart_cost(self, eta_result) -> float:
        """Estimate restart overhead used by REPLAN gating.

        Preference order:
        1) ETA breakdown restart_cost_replan (if finite, >0)
        2) Current RestartCosts model
        3) Conservative fallback (env-configurable)
        """
        breakdown = getattr(eta_result, "costs_breakdown", {}) or {}
        restart_cost = float(breakdown.get("restart_cost_replan", 0.0) or 0.0)
        if math.isfinite(restart_cost) and restart_cost > 0:
            return restart_cost

        costs = self.eta_calculator.restart_costs
        modeled = float(
            costs.C_load
            + costs.D_replan
            + costs.R_replan * costs.T_base
            + costs.T_opt_K
        )
        if math.isfinite(modeled) and modeled > 0:
            return modeled

        return max(0.0, self.replan_fallback_restart_sec)
    
    def _calculate_confidence(self, eta_result, K_rem: int, current_slowdown: float) -> float:
        """결정에 대한 신뢰도 계산 (0.0 ~ 1.0)"""
        eta_values = eta_result.eta_values
        optimal_policy = eta_result.optimal_policy
        
        # 1. ETA 값들의 차이가 클수록 높은 신뢰도
        eta_list = list(eta_values.values())
        eta_range = max(eta_list) - min(eta_list)
        optimal_eta = eta_values[optimal_policy]
        
        if eta_range > 0:
            relative_improvement = eta_range / max(eta_list)
            confidence_base = min(0.9, relative_improvement * 2)  # 최대 90%
        else:
            confidence_base = 0.5  # 차이 없으면 중간 신뢰도
            
        # 2. 조건별 가중치 적용
        confidence = confidence_base
        
        # 후반부일수록 KEEP에 대한 신뢰도 증가
        if self.progress_tracker.is_late_stage():
            if optimal_policy == Policy.KEEP:
                confidence = min(0.95, confidence * 1.2)
            else:
                confidence = confidence * 0.8
        
        # slowdown이 심할수록 REPLAN/DEGRADE에 대한 신뢰도 증가
        if current_slowdown > 1.3:
            if optimal_policy != Policy.KEEP:
                confidence = min(0.95, confidence * 1.1)
                
        # 남은 step이 적을수록 KEEP에 대한 신뢰도 증가  
        if K_rem < 50:
            if optimal_policy == Policy.KEEP:
                confidence = min(0.95, confidence * 1.3)
            else:
                confidence = confidence * 0.7
                
        return max(0.1, min(0.99, confidence))
    
    def _generate_reasoning(self, eta_result, K_rem: int, current_slowdown: float) -> str:
        """결정 논리 설명 생성"""
        optimal_policy = eta_result.optimal_policy
        eta_values = eta_result.eta_values
        
        reasoning_parts = []
        
        # ETA 비교
        eta_keep = eta_values[Policy.KEEP] 
        eta_replan = eta_values[Policy.REPLAN]
        eta_degrade = eta_values[Policy.DEGRADE]
        
        reasoning_parts.append(f"ETA: KEEP={eta_keep:.1f}s, REPLAN={eta_replan:.1f}s, DEGRADE={eta_degrade:.1f}s")
        
        if optimal_policy == Policy.KEEP:
            time_saved = min(eta_replan, eta_degrade) - eta_keep
            reasoning_parts.append(f"Restart overhead ({time_saved:.1f}s) > slowdown cost")
            if K_rem < 100:
                reasoning_parts.append(f"Few remaining steps ({K_rem}) favor KEEP")
                
        elif optimal_policy == Policy.REPLAN:
            time_saved = eta_keep - eta_replan
            reasoning_parts.append(f"REPLAN saves {time_saved:.1f}s vs continuing with slowdown")
            
        elif optimal_policy == Policy.DEGRADE:
            time_saved = eta_keep - eta_degrade
            reasoning_parts.append(f"DEGRADE saves {time_saved:.1f}s, avoiding full restart cost")
            
        # 추가 컨텍스트
        if self.progress_tracker.is_late_stage():
            reasoning_parts.append("Late-stage training")
        if current_slowdown > 1.5:
            reasoning_parts.append("Severe slowdown detected")
            
        return "; ".join(reasoning_parts)
    
    def _create_keep_decision(self, reason: str) -> FailoverDecision:
        """KEEP 정책 결정 생성 (조건 불만족 시)"""
        return FailoverDecision(
            recommended_policy=Policy.KEEP,
            confidence_score=0.9,  # 조건 기반 결정이므로 높은 신뢰도
            eta_analysis={Policy.KEEP: 0.0, Policy.REPLAN: float('inf'), Policy.DEGRADE: float('inf')},
            context_info=self.decision_context.get_context_info(),
            reasoning=reason
        )
    
    def _log_decision(self, decision: FailoverDecision, gpu_id: int, slowdown: float, K_rem: int):
        """결정 과정 로깅"""
        policy_name = decision.recommended_policy.value.upper()
        confidence_pct = decision.confidence_score * 100
        
        self.logger.info(f"🎯 Policy Decision: {policy_name} (confidence: {confidence_pct:.1f}%)")
        self.logger.info(f"   GPU{gpu_id} slowdown: {slowdown:.2f}x, K_rem: {K_rem}, Reasoning: {decision.reasoning}")
        
        # 세부 ETA 분석 로깅
        for policy, eta in decision.eta_analysis.items():
            if isinstance(policy, Policy):
                policy_str = policy.value
            else:
                policy_str = str(policy)
            self.logger.debug(f"   ETA({policy_str}): {eta:.2f}s")
    
    def update_restart_costs(self, measured_costs: RestartCosts):
        """실제 측정된 restart 비용으로 업데이트"""
        self.eta_calculator.update_restart_costs(measured_costs)
        self.logger.info("📊 Restart costs updated with measured values")

    def update_gpu_coefficients(
        self,
        gpu_id: int,
        alpha_comp: Optional[float] = None,
        beta_comm: Optional[float] = None,
    ):
        """Update per-GPU alpha/beta coefficients at runtime."""
        gpu_id = int(gpu_id)
        if alpha_comp is not None:
            self.alpha_g[gpu_id] = float(alpha_comp)
        if beta_comm is not None:
            self.beta_g[gpu_id] = float(beta_comm)
    
    def get_decision_summary(self) -> Dict[str, any]:
        """결정 역사 요약 반환"""
        if not self.decision_history:
            return {"total_decisions": 0}
            
        decisions_by_policy = {}
        total_confidence = 0
        
        for decision in self.decision_history:
            policy = decision.recommended_policy.value
            decisions_by_policy[policy] = decisions_by_policy.get(policy, 0) + 1
            total_confidence += decision.confidence_score
            
        return {
            "total_decisions": len(self.decision_history),
            "decisions_by_policy": decisions_by_policy,
            "avg_confidence": total_confidence / len(self.decision_history),
            "recent_decisions": [
                {
                    "policy": d.recommended_policy.value,
                    "confidence": d.confidence_score,
                    "timestamp": d.decision_timestamp,
                    "reasoning": d.reasoning
                }
                for d in self.decision_history[-5:]  # 최근 5개
            ]
        }