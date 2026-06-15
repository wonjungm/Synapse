"""
Mathematical Model Based Optimizer (Enhanced)
기존 단순 임계치 기반 로직을 ETA 수학적 모델로 대체

수식: ETA(p) = C_restart(p) + K_rem * T(p)
목표: 최적 정책 p* = argmin ETA(p) 선택
"""
import pandas as pd
import numpy as np
import sys
import torch
import time
import json
import os
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import path setup
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Import new mathematical model components
MATHEMATICAL_MODEL_AVAILABLE = False
try:
    # 패키지로 import되는 경우 (python -m benchmarks.soft_target.planner...) 
    from .dynamic_policy_selector import DynamicPolicySelector, Policy
    from .eta_calculator import create_default_restart_costs, RestartCosts
    from .stage_time_predictor import PartitionConfig, StageTimePredictor
    from .progress_tracker import ProgressTracker
    from .dynamic_alpha_beta_estimator import DynamicAlphaBetaEstimator
    MATHEMATICAL_MODEL_AVAILABLE = True
    logger.info("✅ Mathematical model components loaded successfully (relative import)")
except ImportError as e_rel:
    try:
        # 스크립트로 직접 실행되는 경우 (python benchmarks/soft_target/planner/...)
        from planner.dynamic_policy_selector import DynamicPolicySelector, Policy
        from planner.eta_calculator import create_default_restart_costs, RestartCosts
        from planner.stage_time_predictor import PartitionConfig, StageTimePredictor
        from planner.progress_tracker import ProgressTracker
        from planner.dynamic_alpha_beta_estimator import DynamicAlphaBetaEstimator
        MATHEMATICAL_MODEL_AVAILABLE = True
        logger.info("✅ Mathematical model components loaded successfully (absolute import)")
    except ImportError as e_abs:
        MATHEMATICAL_MODEL_AVAILABLE = False
        logger.warning(
            "⚠️ Mathematical model not available, falling back to legacy logic: "
            f"relative={e_rel}; absolute={e_abs}"
        )

class MathematicalFailoverOptimizer:
    """수학적 모델 기반 Failover 최적화"""
    
    def __init__(self, 
                 total_epochs: int = 1,
                 steps_per_epoch: int = 1000,
                 restart_costs: Optional[RestartCosts] = None,
                 baseline_warmup_steps: int = 50,
                 initial_partition_config: Optional[PartitionConfig] = None):
        self.logger = logging.getLogger(f"{__name__}.MathematicalFailoverOptimizer")
        
        # Load profiling data 
        self._load_profiling_data()
        
        # Load GPU performance coefficients
        self._load_gpu_coefficients()

        # Phase-0 baseline collection configuration.
        self.baseline_warmup_steps = max(1, int(baseline_warmup_steps))
        self._baseline_frozen = False
        self._baseline_count_compute: Dict[int, int] = {}
        self._baseline_count_comm: Dict[int, int] = {}
        
        # Initialize mathematical model components  
        if MATHEMATICAL_MODEL_AVAILABLE:
            self.progress_tracker = ProgressTracker(total_epochs, steps_per_epoch)
            self.alpha_beta_estimator = DynamicAlphaBetaEstimator(self.alpha_g, self.beta_g)
            self.policy_selector = DynamicPolicySelector(
                self.progress_tracker,
                restart_costs or create_default_restart_costs(),
                self.alpha_g,
                self.beta_g
            )
            self.use_mathematical_model = True
            self.logger.info("🧠 Using mathematical ETA model for decisions")
        else:
            self.alpha_beta_estimator = None
            self.use_mathematical_model = False
            self.logger.info("📊 Using legacy threshold-based model")
            
        # Legacy state tracking (for fallback)
        self.sustained_time = {gpu_id: 0 for gpu_id in self.alpha_g.keys()}
        self.replan_time = {gpu_id: 0 for gpu_id in self.alpha_g.keys()}
        self.degrade_time = {gpu_id: 0 for gpu_id in self.alpha_g.keys()}
                # Sticky suspect GPU selection for localization stability
        self._suspect_gpu_id: Optional[int] = None
        self._suspect_switch_counter: int = 0
        self._suspect_hold_evals: int = int(os.environ.get("FAILOVER_SUSPECT_HOLD_EVALS", "3"))
        self._suspect_switch_margin: float = float(os.environ.get("FAILOVER_SUSPECT_SWITCH_MARGIN", "0.10"))
        # ✅ Step 1: Current partition tracking - use provided YAML config or fallback to default
        # This ensures optimizer knows the actual runtime partition from day 1
        if initial_partition_config is not None:
            self.current_partition = initial_partition_config
            self.logger.info(f"📌 Initialized current_partition from YAML: snet={initial_partition_config.snet_partition}, tnet={initial_partition_config.tnet_partition}")
        else:
            self.current_partition = self._create_default_partition()
            self.logger.warning("⚠️ Initialized current_partition from default (not from YAML) - consider passing initial_partition_config")

        # Runtime baseline buffers for dynamic alpha/beta estimation
        self._baseline_compute_time: Dict[int, float] = {}
        self._baseline_comm_time: Dict[int, float] = {}
        self._latest_compute_time: Dict[int, float] = {}
        self._latest_comm_time: Dict[int, float] = {}

        # Runtime DP partitioner used at REPLAN/DEGRADE execution time.
        suppress_log = initial_partition_config is not None
        self._runtime_stage_time_predictor = StageTimePredictor(suppress_alpha_beta_log=suppress_log)
        self._sync_runtime_profile_with_partition()

        # Restart-based failover automation hooks (optional).
        self._auto_restart_on_failover = False
        self._checkpoint_saver: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
        self._restart_config_path = os.path.join(os.getcwd(), "restart_config.json")
        self._pending_restart_transition: Optional[Dict[str, Any]] = None

        # ETA breakdown debug log (JSONL)
        self._eta_debug_log_path = os.path.join(
            os.getcwd(),
            "benchmarks",
            "planner",
            "logs",
            "eta_breakdown.jsonl",
        )
        os.makedirs(os.path.dirname(self._eta_debug_log_path), exist_ok=True)
        
        self.logger.info(f"📈 Optimizer initialized: {self.snet_num_layers} SNet + {self.tnet_num_layers} TNet layers")

    def configure_failover_restart(
        self,
        restart_config_path: Optional[str] = None,
        checkpoint_saver: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        auto_restart_on_failover: bool = False,
    ):
        """Configure restart-based failover automation after REPLAN/DEGRADE apply."""
        if restart_config_path:
            self._restart_config_path = restart_config_path
            os.makedirs(os.path.dirname(self._restart_config_path), exist_ok=True)
        self._checkpoint_saver = checkpoint_saver
        self._auto_restart_on_failover = bool(auto_restart_on_failover)
        self.logger.info(
            "⚙️ Failover restart automation configured "
            f"(enabled={self._auto_restart_on_failover}, restart_config={self._restart_config_path})"
        )

    def _partition_boundaries(self, lengths: List[int]) -> List[List[int]]:
        """Convert partition lengths to [start, end) boundaries."""
        boundaries: List[List[int]] = []
        start = 0
        for length in lengths:
            end = start + int(length)
            boundaries.append([start, end])
            start = end
        return boundaries

    def _partition_to_payload(self, partition: Optional[PartitionConfig]) -> Optional[Dict[str, Any]]:
        if partition is None:
            return None
        return {
            "gpu_assignment": [int(g) for g in partition.gpu_assignment],
            "snet_partition": [int(v) for v in partition.snet_partition],
            "tnet_partition": [int(v) for v in partition.tnet_partition],
            "snet_stage_boundaries": self._partition_boundaries(partition.snet_partition),
            "tnet_stage_boundaries": self._partition_boundaries(partition.tnet_partition),
        }

    def estimate_partition_nominal_step_time(self, partition: Optional[PartitionConfig]) -> float:
        """Estimate partition bottleneck under neutral (healthy) GPU coefficients."""
        if partition is None:
            return float("inf")
        neutral_alpha = {int(g): 1.0 for g in partition.gpu_assignment}
        neutral_beta = {int(g): 1.0 for g in partition.gpu_assignment}
        return float(
            self._runtime_stage_time_predictor.calculate_partition_bottleneck_time(
                partition,
                neutral_alpha,
                neutral_beta,
            )
        )

    def export_restart_state(self) -> Dict[str, Any]:
        """Serialize optimizer runtime baseline state for failover restart."""
        state: Dict[str, Any] = {
            "baseline_warmup_steps": int(self.baseline_warmup_steps),
            "baseline_frozen": bool(self._baseline_frozen),
            "baseline_count_compute": {
                int(k): int(v) for k, v in self._baseline_count_compute.items()
            },
            "baseline_count_comm": {
                int(k): int(v) for k, v in self._baseline_count_comm.items()
            },
            "baseline_compute_time": {
                int(k): float(v) for k, v in self._baseline_compute_time.items()
            },
            "baseline_comm_time": {
                int(k): float(v) for k, v in self._baseline_comm_time.items()
            },
            "latest_compute_time": {
                int(k): float(v) for k, v in self._latest_compute_time.items()
            },
            "latest_comm_time": {
                int(k): float(v) for k, v in self._latest_comm_time.items()
            },
            "suspect_gpu_id": None if self._suspect_gpu_id is None else int(self._suspect_gpu_id),
            "suspect_switch_counter": int(self._suspect_switch_counter),
        }
        if self.use_mathematical_model:
            state["progress_step"] = int(self.progress_tracker.progress.current_step)
            state["progress_epoch"] = int(self.progress_tracker.progress.current_epoch)
        return state

    def restore_restart_state(self, state: Optional[Dict[str, Any]]) -> bool:
        """Restore optimizer baseline/runtime state after failover restart."""
        if not isinstance(state, dict):
            return False

        self.baseline_warmup_steps = max(
            1,
            int(state.get("baseline_warmup_steps", self.baseline_warmup_steps) or self.baseline_warmup_steps),
        )
        self._baseline_frozen = bool(state.get("baseline_frozen", self._baseline_frozen))
        self._baseline_count_compute = {
            int(k): int(v) for k, v in (state.get("baseline_count_compute") or {}).items()
        }
        self._baseline_count_comm = {
            int(k): int(v) for k, v in (state.get("baseline_count_comm") or {}).items()
        }
        self._baseline_compute_time = {
            int(k): float(v) for k, v in (state.get("baseline_compute_time") or {}).items()
        }
        self._baseline_comm_time = {
            int(k): float(v) for k, v in (state.get("baseline_comm_time") or {}).items()
        }
        self._latest_compute_time = {
            int(k): float(v) for k, v in (state.get("latest_compute_time") or {}).items()
        }
        self._latest_comm_time = {
            int(k): float(v) for k, v in (state.get("latest_comm_time") or {}).items()
        }
        suspect_gpu = state.get("suspect_gpu_id")
        self._suspect_gpu_id = None if suspect_gpu is None else int(suspect_gpu)
        self._suspect_switch_counter = int(state.get("suspect_switch_counter", 0) or 0)

        if self.use_mathematical_model:
            progress_step = int(state.get("progress_step", self.progress_tracker.progress.current_step) or 0)
            progress_epoch = int(state.get("progress_epoch", self.progress_tracker.progress.current_epoch) or 0)
            self.progress_tracker.update_step(progress_step, progress_epoch)
            if self._baseline_frozen:
                self._refresh_restart_cost_model()

        self.logger.info(
            "✅ Restored optimizer baseline state from checkpoint "
            f"(baseline_frozen={self._baseline_frozen}, compute_gpus={sorted(self._baseline_compute_time.keys())})"
        )
        return True

    def _build_restart_payload(self, policy: str) -> Dict[str, Any]:
        """Build restart config payload from the latest partition/coefficient state."""
        step_id = 0
        t_base = 0.0
        c_load = 4.37
        d_replan = 14.0
        r_replan = 50.0
        if self.use_mathematical_model:
            step_id = int(self.progress_tracker.progress.current_step)
            costs = self.policy_selector.eta_calculator.restart_costs
            t_base = float(costs.T_base)
            c_load = float(costs.C_load)
            d_replan = float(costs.D_replan)
            r_replan = float(costs.R_replan)

        active_gpus = [int(g) for g in self.current_partition.gpu_assignment]
        alpha_to_save = {
            int(gpu_id): float(self.alpha_g.get(int(gpu_id), 1.0))
            for gpu_id in active_gpus
        }
        beta_to_save = {
            int(gpu_id): float(self.beta_g.get(int(gpu_id), 1.0))
            for gpu_id in active_gpus
        }

        payload: Dict[str, Any] = {
            "timestamp": time.time(),
            "trigger_policy": str(policy),
            "step_id": step_id,
            "partition": self._partition_to_payload(self.current_partition),
            "alpha_comp": alpha_to_save,
            "beta_comm": beta_to_save,
            "restart_overhead": {
                "formula": "C_load + D_replan + R_replan * T_base",
                "t_base": t_base,
                "value": float(c_load + d_replan + r_replan * t_base),
            },
        }
        if self._pending_restart_transition is not None:
            payload["partition_transition"] = deepcopy(self._pending_restart_transition)
        return payload

    def _write_restart_config(self, payload: Dict[str, Any]):
        """Overwrite restart_config.json with latest failover partition config."""
        with open(self._restart_config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _trigger_failover_restart(self, policy: str):
        # 재시작을 위한 정보 저장
        if policy not in {"REPLAN", "DEGRADE"}:
            return
        if not self._auto_restart_on_failover:
            return

        payload = self._build_restart_payload(policy)
        checkpoint_path = None

        self._write_restart_config(payload)

        if self._checkpoint_saver is not None:
            try:
                checkpoint_path = self._checkpoint_saver(payload)
            except Exception as e:
                self.logger.error(f"❌ Failed to save failover checkpoint: {e}")
                checkpoint_path = None

        if not checkpoint_path:
            # checkpoint 없이 재시작하면 같은 failover가 반복될 수 있으므로 무한 재시작을 막음.
            self.logger.error(
                "❌ Restart skipped: checkpoint artifact was not created. "
                "Continuing in-process to avoid restart loop."
            )
            return

        payload["checkpoint_path"] = str(checkpoint_path)
        self._write_restart_config(payload)
        self._pending_restart_transition = None

        self.logger.error(
            f"Failover triggered (Policy: {policy}). "
            f"Partition saved to {os.path.basename(self._restart_config_path)}. "
            "Restarting process..."
        )
        raise SystemExit(42)
    
    def _load_profiling_data(self):
        """프로파일링 데이터 로드"""
        # Load CSV files
        tnet_df = pd.read_csv('./benchmarks/soft_target/planner/profile/tnet.csv')
        snet_df = pd.read_csv('./benchmarks/soft_target/planner/profile/snet.csv')
        self.tnet_num_layers = len(tnet_df)
        self.snet_num_layers = len(snet_df)
        
        # Extract profiling data
        self.tnet_forward_times_ms = tnet_df['forward_time_ms'].tolist()
        self.tnet_param_sizes_kb = tnet_df['parameter_size_kb'].tolist()
        self.tnet_input_activation_sizes_kb = tnet_df['input_activation_size_kb'].tolist()
        
        self.snet_forward_times_ms = snet_df['forward_time_ms'].tolist()
        self.snet_backward_times_ms = snet_df['backward_time_ms'].tolist()
        self.snet_param_sizes_kb = snet_df['parameter_size_kb'].tolist()
        self.snet_input_activation_sizes_kb = snet_df['input_activation_size_kb'].tolist()
        
        # Network configuration
        self.bandwidth_gbps = 8
        self.bandwidth_kbps = self.bandwidth_gbps * 1024 * 1024
        self.num_stages = 4
        
        # GPU memory info
        device = torch.device("cuda:0")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        self.total_memory_kb = total_memory / 1024

    def _resample_profile_vector(self, values: List[float], target_len: int) -> List[float]:
        """Resample a per-layer profile vector to match runtime layer count."""
        if target_len <= 0:
            return []
        if not values:
            return [0.0] * target_len
        if len(values) == target_len:
            return list(values)
        if len(values) == 1:
            return [float(values[0])] * target_len

        x_old = np.linspace(0.0, 1.0, len(values))
        x_new = np.linspace(0.0, 1.0, target_len)
        return np.interp(x_new, x_old, np.asarray(values, dtype=float)).tolist()

    def _sync_runtime_profile_with_partition(self):
        """Align profiler layer dimensions with runtime partition (e.g., 57/30 models)."""
        target_snet_layers = int(sum(self.current_partition.snet_partition))
        target_tnet_layers = int(sum(self.current_partition.tnet_partition))
        if target_snet_layers <= 0 or target_tnet_layers <= 0:
            self.logger.warning(
                "⚠️ Skip runtime profile sync due to invalid partition sizes: "
                f"snet={target_snet_layers}, tnet={target_tnet_layers}"
            )
            return

        if (
            target_snet_layers == self.snet_num_layers
            and target_tnet_layers == self.tnet_num_layers
        ):
            return

        self.logger.warning(
            "⚠️ Profile/partition layer mismatch detected. "
            f"Resampling profile vectors: snet {self.snet_num_layers}->{target_snet_layers}, "
            f"tnet {self.tnet_num_layers}->{target_tnet_layers}"
        )

        self.snet_forward_times_ms = self._resample_profile_vector(self.snet_forward_times_ms, target_snet_layers)
        self.snet_backward_times_ms = self._resample_profile_vector(self.snet_backward_times_ms, target_snet_layers)
        self.snet_param_sizes_kb = self._resample_profile_vector(self.snet_param_sizes_kb, target_snet_layers)
        self.snet_input_activation_sizes_kb = self._resample_profile_vector(self.snet_input_activation_sizes_kb, target_snet_layers)

        self.tnet_forward_times_ms = self._resample_profile_vector(self.tnet_forward_times_ms, target_tnet_layers)
        self.tnet_param_sizes_kb = self._resample_profile_vector(self.tnet_param_sizes_kb, target_tnet_layers)
        self.tnet_input_activation_sizes_kb = self._resample_profile_vector(self.tnet_input_activation_sizes_kb, target_tnet_layers)

        self.snet_num_layers = target_snet_layers
        self.tnet_num_layers = target_tnet_layers

        predictor = self._runtime_stage_time_predictor
        predictor.snet_num_layers = target_snet_layers
        predictor.tnet_num_layers = target_tnet_layers
        predictor.snet_forward_times_ms = self._resample_profile_vector(
            predictor.snet_forward_times_ms,
            target_snet_layers,
        )
        predictor.snet_backward_times_ms = self._resample_profile_vector(
            predictor.snet_backward_times_ms,
            target_snet_layers,
        )
        predictor.snet_param_sizes_kb = self._resample_profile_vector(
            predictor.snet_param_sizes_kb,
            target_snet_layers,
        )
        predictor.snet_input_activation_sizes_kb = self._resample_profile_vector(
            predictor.snet_input_activation_sizes_kb,
            target_snet_layers,
        )
        predictor.snet_output_activation_sizes_kb = self._resample_profile_vector(
            predictor.snet_output_activation_sizes_kb,
            target_snet_layers,
        )
        predictor.tnet_forward_times_ms = self._resample_profile_vector(
            predictor.tnet_forward_times_ms,
            target_tnet_layers,
        )
        predictor.tnet_param_sizes_kb = self._resample_profile_vector(
            predictor.tnet_param_sizes_kb,
            target_tnet_layers,
        )
        predictor.tnet_input_activation_sizes_kb = self._resample_profile_vector(
            predictor.tnet_input_activation_sizes_kb,
            target_tnet_layers,
        )

        self.logger.info(
            "✅ Runtime profile synchronized with partition: "
            f"SNet={self.snet_num_layers}, TNet={self.tnet_num_layers}"
        )
        
    def _load_gpu_coefficients(self):
        """Initialize ratio-only coefficients (Phase-0 starts from homogeneous defaults)."""
        gpu_count = max(1, torch.cuda.device_count())
        self.alpha_g = {gpu_id: 1.0 for gpu_id in range(gpu_count)}
        self.beta_g = {gpu_id: 1.0 for gpu_id in range(gpu_count)}
        self.logger.info("📊 Initialized ratio-only GPU coefficients: alpha_g=1.0, beta_g=1.0")

    def _is_phase0_baseline_active(self) -> bool:
        if self._baseline_frozen:
            return False
        if not self.use_mathematical_model:
            return False
        return self.progress_tracker.progress.current_step < self.baseline_warmup_steps

    def _try_freeze_phase0_baseline(self):
        if self._baseline_frozen or not self.use_mathematical_model:
            return
        if self.progress_tracker.progress.current_step < self.baseline_warmup_steps:
            return
        self._baseline_frozen = True
        self._refresh_restart_cost_model()
        self.logger.info(
            f"📌 Phase-0 baseline frozen at step={self.progress_tracker.progress.current_step} "
            f"(warmup={self.baseline_warmup_steps})"
        )

    def _estimate_baseline_step_time(self) -> float:
        """Estimate baseline step bottleneck time from collected per-GPU runtime timings."""
        candidates = []
        gpu_ids = set(self._baseline_compute_time.keys()) | set(self._baseline_comm_time.keys())
        for gpu_id in gpu_ids:
            comp = float(self._baseline_compute_time.get(gpu_id, 0.0) or 0.0)
            comm = float(self._baseline_comm_time.get(gpu_id, 0.0) or 0.0)
            total = comp + comm
            if total > 0:
                candidates.append(total)

        if not candidates:
            return 1.0
        return max(candidates)

    def _refresh_restart_cost_model(self):
        """Apply experiment-0B restart constants with runtime baseline step time."""
        if not self.use_mathematical_model:
            return

        t_base = self._estimate_baseline_step_time()

        # Guardrail: keep T_base in a realistic step-time range (seconds).
        # Runtime profiler summaries can occasionally produce outlier scales,
        # which would explode restart ETA (R * T_base) and force KEEP always.
        force_t_base = os.environ.get("FAILOVER_FORCE_T_BASE_SEC", "").strip()
        if force_t_base:
            try:
                t_base = float(force_t_base)
            except ValueError:
                self.logger.warning(f"Invalid FAILOVER_FORCE_T_BASE_SEC={force_t_base}, ignoring")
        t_base = max(0.05, min(5.0, float(t_base)))

        # Conservative defaults from observed restart downtime (~28-30s).
        # Keep env overrides for low-risk tuning during paper prep.
        c_load = float(os.environ.get("FAILOVER_RESTART_C_LOAD_SEC", "4.37"))
        d_replan = float(os.environ.get("FAILOVER_RESTART_D_REPLAN_SEC", "14.0"))
        d_degrade = float(os.environ.get("FAILOVER_RESTART_D_DEGRADE_SEC", "10.0"))
        r_replan = float(os.environ.get("FAILOVER_RESTART_R_REPLAN", "50.0"))
        r_degrade = float(os.environ.get("FAILOVER_RESTART_R_DEGRADE", "50.0"))

        updated = RestartCosts(
            C_load=c_load,
            D_replan=d_replan,
            D_degrade=d_degrade,
            R_replan=r_replan,
            R_degrade=r_degrade,
            T_base=t_base,
            T_opt_K=0.0,
            T_opt_K_minus_1=0.0,
        )
        self.policy_selector.update_restart_costs(updated)

    def _estimate_current_step_time(self) -> float:
        """Estimate current step bottleneck from latest per-GPU compute+comm timings."""
        candidates = []
        gpu_ids = set(self._latest_compute_time.keys()) | set(self._latest_comm_time.keys())
        for gpu_id in gpu_ids:
            comp = float(self._latest_compute_time.get(gpu_id, 0.0) or 0.0)
            comm = float(self._latest_comm_time.get(gpu_id, 0.0) or 0.0)
            total = comp + comm
            if total > 0:
                candidates.append(total)
        if not candidates:
            return 0.0
        return max(candidates)

    def _write_eta_breakdown_jsonl(
        self,
        gpu_id: int,
        current_slowdown: float,
        recommended_policy: str,
        eta_keep: float,
        eta_replan: float,
        eta_degrade: float,
        reasoning: str,
    ):
        """Persist ETA breakdown for auditability of policy decisions."""
        if not self.use_mathematical_model:
            return

        costs = self.policy_selector.eta_calculator.restart_costs
        t_base = float(costs.T_base)
        replan_overhead = costs.C_load + costs.D_replan + costs.R_replan * t_base

        # DEGRADE 정책: 현재 gpu_assignment과 일치하는 alpha/beta만 저장
        alpha_to_save = self.alpha_g
        beta_to_save = self.beta_g
        if recommended_policy == "DEGRADE" and self.current_partition is not None:
            # DEGRADE 시: 남은 GPU들의 alpha/beta만 저장
            alpha_to_save = {gpu: self.alpha_g.get(gpu, 0.5) 
                           for gpu in self.current_partition.gpu_assignment}
            beta_to_save = {gpu: self.beta_g.get(gpu, 1.0) 
                          for gpu in self.current_partition.gpu_assignment}
            self.logger.error(f"🔧 DEGRADE: Filtered alpha/beta to {list(alpha_to_save.keys())}")
        
        payload = {
            "timestamp": time.time(),
            "step_id": int(self.progress_tracker.progress.current_step),
            "gpu_id": int(gpu_id),
            "slowdown": float(current_slowdown),
            "recommended_policy": str(recommended_policy),
            "alpha_comp": {int(k): float(v) for k, v in alpha_to_save.items()},
            "beta_comm": {int(k): float(v) for k, v in beta_to_save.items()},
            "current_step_time": float(self._estimate_current_step_time()),
            "t_base": t_base,
            "eta": {
                "keep": float(eta_keep),
                "replan": float(eta_replan),
                "degrade": float(eta_degrade),
            },
            "eta_compare_keep_vs_replan": {
                "keep": float(eta_keep),
                "replan": float(eta_replan),
                "delta_keep_minus_replan": float(eta_keep - eta_replan),
            },
            "replan_overhead": {
                "formula": "C_load + D_replan + R_replan * T_base",
                "value": float(replan_overhead),
                "c_load": float(costs.C_load),
                "d_replan": float(costs.D_replan),
                "r_replan": float(costs.R_replan),
                "t_base": t_base,
            },
            "reasoning": reasoning,
        }

        try:
            with open(self._eta_debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write ETA debug JSONL: {e}")
    
    def _create_default_partition(self) -> PartitionConfig:
        """기본 파티션 구성 생성"""
        # 균등 분할로 시작
        snet_layers_per_stage = self.snet_num_layers // self.num_stages
        tnet_layers_per_stage = self.tnet_num_layers // self.num_stages
        
        return PartitionConfig(
            snet_partition=[snet_layers_per_stage] * self.num_stages,
            tnet_partition=[tnet_layers_per_stage] * self.num_stages,
            gpu_assignment=list(range(self.num_stages))
        )
    
    def update_training_progress(self, step_id: int, epoch: int = 0):
        """훈련 진행상황 업데이트 (수학적 모델에서 K_rem 계산에 사용)"""
        if self.use_mathematical_model:
            self.progress_tracker.update_step(step_id, epoch)
    
    def evaluate_slowdown_and_decide(self, 
                                   gpu_id: int, 
                                   current_slowdown: float,
                                   failed_gpus: Optional[List[int]] = None,
                                   trigger_confirmed: bool = False,) -> str:
        """
        성능 저하 감지 시 최적 정책 결정
        
        Args:
            gpu_id: 성능 저하가 감지된 GPU ID
            current_slowdown: 현재 slowdown 비율
            failed_gpus: 장애가 발생한 GPU 목록
            
        Returns:
            str: 추천 정책 ("KEEP", "REPLAN", "DEGRADE")
        """
        if self.use_mathematical_model:
            return self._decide_with_mathematical_model(gpu_id, current_slowdown, failed_gpus, trigger_confirmed=trigger_confirmed,)
        else:
            return self._decide_with_legacy_logic(gpu_id, current_slowdown)
    
    def _decide_with_mathematical_model(self, 
                                      gpu_id: int, 
                                      current_slowdown: float,
                                      failed_gpus: Optional[List[int]] = None,
                                      trigger_confirmed: bool = False,) -> str:
        self._try_freeze_phase0_baseline()
        if self._is_phase0_baseline_active():
            self.logger.info(
                f"🧪 Phase-0 baseline collection active "
                f"(step={self.progress_tracker.progress.current_step}/{self.baseline_warmup_steps}) -> KEEP"
            )
            return "KEEP"
        try:
            decision = self.policy_selector.evaluate_slowdown(
                gpu_id, 
                current_slowdown, 
                self.current_partition,
                failed_gpus,
                trigger_confirmed=trigger_confirmed,
            )
            
            policy_name = decision.recommended_policy.value.upper()
            confidence = decision.confidence_score * 100
            
            self.logger.info(f"🎯 Mathematical Model Decision: {policy_name} (confidence: {confidence:.1f}%)")
            self.logger.info(f"   Reasoning: {decision.reasoning}")

            eta_keep = float(decision.eta_analysis.get(Policy.KEEP, float("inf")))
            eta_replan = float(decision.eta_analysis.get(Policy.REPLAN, float("inf")))
            eta_degrade = float(decision.eta_analysis.get(Policy.DEGRADE, float("inf")))
            self._write_eta_breakdown_jsonl(
                gpu_id=gpu_id,
                current_slowdown=current_slowdown,
                recommended_policy=policy_name,
                eta_keep=eta_keep,
                eta_replan=eta_replan,
                eta_degrade=eta_degrade,
                reasoning=decision.reasoning,
            )
            
            return policy_name
            
        except Exception as e:
            self.logger.error(f"❌ Mathematical model failed, falling back to legacy: {e}")
            return self._decide_with_legacy_logic(gpu_id, current_slowdown)
    
    def _decide_with_legacy_logic(self, gpu_id: int, current_slowdown: float) -> str:
        """기존 임계치 기반 결정 (fallback)"""
        if current_slowdown < 1.1:
            self.sustained_time[gpu_id] = 0
            self.replan_time[gpu_id] = 0
            self.degrade_time[gpu_id] = 0
            return "KEEP"

        if current_slowdown >= 1.25:
            self.replan_time[gpu_id] += 1
            self.sustained_time[gpu_id] += 1
            if self.replan_time[gpu_id] >= 30:
                self.logger.info(f"📊 Legacy Model: DEGRADE triggered for GPU {gpu_id}")
                self.degrade_time[gpu_id] = 0
                return "DEGRADE"

        if current_slowdown >= 1.1:
            self.sustained_time[gpu_id] += 1
            if self.sustained_time[gpu_id] >= 10:
                self.logger.info(f"📊 Legacy Model: REPLAN triggered for GPU {gpu_id}")
                self.replan_time[gpu_id] = 0
                self.degrade_time[gpu_id] = 0
                return "REPLAN"

        return "KEEP"
    
    def execute_policy(self, policy: str, gpu_id: int, current_slowdown: float):
        """선택된 정책 실행"""
        if policy == "REPLAN":
            self._execute_replan(gpu_id, current_slowdown)
        elif policy == "DEGRADE":
            self._execute_degrade(gpu_id, current_slowdown)
        elif policy == "KEEP":
            self._execute_keep(gpu_id, current_slowdown)
    
    def _execute_replan(self, gpu_id: int, slowdown: float):
        """REPLAN 정책 실행"""
        self.logger.info(f"🔄 Executing REPLAN for GPU {gpu_id} (slowdown: {slowdown:.2f})")

        previous_partition = deepcopy(self.current_partition)
        active_gpus = list(self.current_partition.gpu_assignment)
        new_partition = self.replan_optimizer(
            "tnet.csv",
            "snet.csv",
            slowdown,
            affected_gpu=gpu_id,
            gpu_assignment=active_gpus,
        )
        if new_partition is not None:
            self._pending_restart_transition = {
                "policy": "REPLAN",
                "previous_partition": self._partition_to_payload(previous_partition),
                "previous_nominal_step_time": self.estimate_partition_nominal_step_time(previous_partition),
                "new_nominal_step_time": self.estimate_partition_nominal_step_time(new_partition),
            }
            self.current_partition = new_partition
            self.logger.info(f"🔁 REPLAN applied new partition: {self.current_partition}")
            self._trigger_failover_restart("REPLAN")
        else:
            self.logger.warning("REPLAN failed to produce a valid DP partition")
        
    def _execute_degrade(self, gpu_id: int, slowdown: float):
        """DEGRADE 정책 실행"""
        self.logger.info(f"⬇️ Executing DEGRADE by excluding GPU {gpu_id} (slowdown: {slowdown:.2f})")

        previous_partition = deepcopy(self.current_partition)
        active_gpus = [g for g in self.current_partition.gpu_assignment if g != gpu_id]
        if not active_gpus:
            self.logger.warning("DEGRADE skipped: no active GPU remains after exclusion")
            return

        new_partition = self._run_realtime_dp_repartition(active_gpus, policy_name="DEGRADE")
        if new_partition is None:
            self.logger.warning("DEGRADE failed to produce a valid DP partition")
            return

        self._pending_restart_transition = {
            "policy": "DEGRADE",
            "previous_partition": self._partition_to_payload(previous_partition),
            "previous_nominal_step_time": self.estimate_partition_nominal_step_time(previous_partition),
            "new_nominal_step_time": self.estimate_partition_nominal_step_time(new_partition),
        }
        self.current_partition = new_partition
        self.logger.info(f"⬇️ DEGRADE applied new partition: {self.current_partition}")
        self._trigger_failover_restart("DEGRADE")
        
    def _execute_keep(self, gpu_id: int, slowdown: float):
        """KEEP 정책 실행 (아무것도 하지 않음)"""
        self.logger.info(f"✅ Executing KEEP for GPU {gpu_id} (slowdown: {slowdown:.2f}) - No action needed")
    
    def get_gpu_slowdown(self, gpu_id: int) -> float:
        return max(
            float(self.alpha_g.get(int(gpu_id), 1.0)),
            float(self.beta_g.get(int(gpu_id), 1.0)),
        )

    def identify_slow_gpu(self, preferred_gpu: Optional[int] = None) -> int:
        """
        Stable suspect-GPU localization.

        - synthetic 실험이면 preferred_gpu를 우선 사용
        - 그 외에는 sticky suspect + hysteresis로 순간 흔들림을 완화
        """
        candidate_gpu_ids = set(self.alpha_g.keys()) | set(self.beta_g.keys())
        if self.current_partition is not None and self.current_partition.gpu_assignment:
            active = set(int(g) for g in self.current_partition.gpu_assignment)
            candidate_gpu_ids &= active

        if not candidate_gpu_ids:
            return 0

        slowdown_by_gpu = {
            int(gpu_id): self.get_gpu_slowdown(int(gpu_id))
            for gpu_id in sorted(candidate_gpu_ids)
        }

        # Synthetic experiment prior: keep the injected GPU as suspect.
        if preferred_gpu is not None and int(preferred_gpu) in slowdown_by_gpu:
            chosen = int(preferred_gpu)
            self._suspect_gpu_id = chosen
            self._suspect_switch_counter = 0
            self.logger.info(
                f"🔍 Identified slow GPU (preferred prior): GPU {chosen} "
                f"with slowdown {slowdown_by_gpu[chosen]:.3f}x"
            )
            return chosen

        best_gpu_id, best_slowdown = max(
            slowdown_by_gpu.items(),
            key=lambda kv: (kv[1], -kv[0]),
        )

        if self._suspect_gpu_id is None or self._suspect_gpu_id not in slowdown_by_gpu:
            self._suspect_gpu_id = best_gpu_id
            self._suspect_switch_counter = 0
            self.logger.info(
                f"🔍 Identified slow GPU (initial): GPU {best_gpu_id} "
                f"with slowdown {best_slowdown:.3f}x"
            )
            return best_gpu_id

        current_suspect = int(self._suspect_gpu_id)
        current_suspect_slowdown = slowdown_by_gpu[current_suspect]

        # If challenger is not clearly better, keep current suspect.
        if (
            best_gpu_id == current_suspect
            or best_slowdown <= current_suspect_slowdown + self._suspect_switch_margin
        ):
            self._suspect_switch_counter = 0
            chosen = current_suspect
        else:
            self._suspect_switch_counter += 1
            if self._suspect_switch_counter >= self._suspect_hold_evals:
                self._suspect_gpu_id = best_gpu_id
                self._suspect_switch_counter = 0
                chosen = best_gpu_id
            else:
                chosen = current_suspect

        self.logger.info(
            f"🔍 Identified slow GPU: GPU {chosen} "
            f"(best={best_gpu_id}:{best_slowdown:.3f}x, "
            f"suspect={self._suspect_gpu_id}, "
            f"hold_counter={self._suspect_switch_counter})"
        )
        return chosen
    
    def _run_realtime_dp_repartition(self, gpu_assignment: List[int], policy_name: str) -> Optional[PartitionConfig]:
        """Run minimax contiguous DP now using the latest monitored alpha/beta."""
        if not gpu_assignment:
            return None

        partition = self._runtime_stage_time_predictor.solve_optimal_partition(
            gpu_ids=list(gpu_assignment),
            alpha_g=self.alpha_g,
            beta_g=self.beta_g,
        )
        if partition is None:
            return None

        bottleneck = self._runtime_stage_time_predictor.calculate_partition_bottleneck_time(
            partition,
            alpha_g=self.alpha_g,
            beta_g=self.beta_g,
        )
        self.logger.info(
            f"🧮 {policy_name} realtime DP repartition complete "
            f"(bottleneck={bottleneck:.4f}s, alpha={self.alpha_g}, beta={self.beta_g})"
        )
        return partition

    def replan_optimizer(
        self,
        tnet_csv: str,
        snet_csv: str,
        slowdown: float,
        avail_mem=None,
        affected_gpu: Optional[int] = None,
        gpu_assignment: Optional[List[int]] = None,
    ):
        """Real-time DP repartitioner for REPLAN using latest runtime coefficients."""
        self.logger.info(f"🔧 Running realtime DP replan (slowdown factor: {slowdown})")
        del tnet_csv, snet_csv, avail_mem, affected_gpu  # Kept for backward-compatible signature.
        assignment = list(gpu_assignment) if gpu_assignment is not None else list(self.current_partition.gpu_assignment)
        return self._run_realtime_dp_repartition(assignment, policy_name="REPLAN")
    
    def get_current_performance_summary(self) -> Dict[str, any]:
        """현재 성능 및 결정 상태 요약"""
        summary = {
            "model_type": "mathematical" if self.use_mathematical_model else "legacy",
            "snet_layers": self.snet_num_layers,
            "tnet_layers": self.tnet_num_layers,
            "num_stages": self.num_stages,
            "alpha_g": dict(self.alpha_g),
            "beta_g": dict(self.beta_g),
        }
        
        if self.use_mathematical_model:
            summary.update({
                "progress_info": self.progress_tracker.get_progress_info(),
                "decision_summary": self.policy_selector.get_decision_summary()
            })
        else:
            summary.update({
                "sustained_time": dict(self.sustained_time),
                "replan_time": dict(self.replan_time),
                "degrade_time": dict(self.degrade_time)
            })
            
        return summary
    
    def update_measured_costs(self, measured_costs: Dict[str, float]):
        """실제 측정된 비용으로 수학적 모델 업데이트"""
        if not self.use_mathematical_model:
            return
            
        # 측정된 비용을 RestartCosts 객체로 변환
        restart_costs = RestartCosts(
            C_load=measured_costs.get('checkpoint_load_time', 4.37),
            D_replan=measured_costs.get('replan_time', 14.0),
            D_degrade=measured_costs.get('degrade_time', 10.0),
            R_replan=measured_costs.get('replan_factor', 50.0),
            R_degrade=measured_costs.get('degrade_factor', 50.0),
            T_base=measured_costs.get('base_stage_time', 1.0),
            T_opt_K=measured_costs.get('optimization_K_time', 0.0),
            T_opt_K_minus_1=measured_costs.get('optimization_K_minus_1_time', 0.0)
        )
        
        self.policy_selector.update_restart_costs(restart_costs)
        self.logger.info("📊 Mathematical model updated with measured costs")

    def update_dynamic_alpha_beta(
        self,
        gpu_id: int,
        current_compute_time: Optional[float] = None,
        baseline_compute_time: Optional[float] = None,
        current_comm_time: Optional[float] = None,
        baseline_comm_time: Optional[float] = None,
        ema: float = 0.4,
    ):
        """Update alpha/beta from measured runtime ratios (optional runtime hook)."""
        if not self.use_mathematical_model or self.alpha_beta_estimator is None:
            return

        compute_ratio = None
        if current_compute_time is not None and baseline_compute_time and baseline_compute_time > 0:
            compute_ratio = float(current_compute_time) / float(baseline_compute_time)

        comm_ratio = None
        if current_comm_time is not None and baseline_comm_time and baseline_comm_time > 0:
            comm_ratio = float(current_comm_time) / float(baseline_comm_time)

        estimate = self.alpha_beta_estimator.update_from_ratios(
            gpu_id=gpu_id,
            compute_ratio=compute_ratio,
            comm_ratio=comm_ratio,
            ema=ema,
        )
        self.policy_selector.update_gpu_coefficients(
            gpu_id=gpu_id,
            alpha_comp=estimate.alpha_comp,
            beta_comm=estimate.beta_comm,
        )
        # Keep local copies in sync for visibility and future fallback logic.
        self.alpha_g[gpu_id] = estimate.alpha_comp
        self.beta_g[gpu_id] = estimate.beta_comm

    def ingest_runtime_timing(
        self,
        gpu_id: int,
        compute_time: float,
        comm_time: float,
        baseline_compute_time: Optional[float] = None,
        baseline_comm_time: Optional[float] = None,
        ema: float = 0.4,
    ):
        """
        Ingest measured per-GPU timings and update alpha/beta.

        If baselines are not provided, first observed values are used as defaults.
        """
        gpu_id = int(gpu_id)
        self._latest_compute_time[gpu_id] = float(compute_time)
        self._latest_comm_time[gpu_id] = float(comm_time)
        self._try_freeze_phase0_baseline()

        # During phase-0, collect baseline only and keep alpha/beta fixed to 1.0.
        if self._is_phase0_baseline_active():
            if compute_time > 0:
                c = self._baseline_count_compute.get(gpu_id, 0)
                prev = self._baseline_compute_time.get(gpu_id, 0.0)
                self._baseline_compute_time[gpu_id] = (prev * c + float(compute_time)) / (c + 1)
                self._baseline_count_compute[gpu_id] = c + 1
            if comm_time > 0:
                c = self._baseline_count_comm.get(gpu_id, 0)
                prev = self._baseline_comm_time.get(gpu_id, 0.0)
                self._baseline_comm_time[gpu_id] = (prev * c + float(comm_time)) / (c + 1)
                self._baseline_count_comm[gpu_id] = c + 1
            # Freeze coefficients to neutral values until baseline collection ends.
            self.alpha_g[gpu_id] = 1.0
            self.beta_g[gpu_id] = 1.0
            if self.use_mathematical_model:
                self.policy_selector.update_gpu_coefficients(gpu_id=gpu_id, alpha_comp=1.0, beta_comm=1.0)
            return

        # Initialize baselines lazily if caller does not provide explicit values.
        if baseline_compute_time is None:
            baseline_compute_time = self._baseline_compute_time.get(gpu_id)
            if baseline_compute_time is None and compute_time > 0:
                baseline_compute_time = float(compute_time)
                self._baseline_compute_time[gpu_id] = baseline_compute_time
        else:
            self._baseline_compute_time[gpu_id] = float(baseline_compute_time)

        if baseline_comm_time is None:
            baseline_comm_time = self._baseline_comm_time.get(gpu_id)
            if baseline_comm_time is None and comm_time > 0:
                baseline_comm_time = float(comm_time)
                self._baseline_comm_time[gpu_id] = baseline_comm_time
        else:
            self._baseline_comm_time[gpu_id] = float(baseline_comm_time)

        self.update_dynamic_alpha_beta(
            gpu_id=gpu_id,
            current_compute_time=compute_time,
            baseline_compute_time=baseline_compute_time,
            current_comm_time=comm_time,
            baseline_comm_time=baseline_comm_time,
            ema=ema,
        )
        if self._baseline_frozen:
            self._refresh_restart_cost_model()

    def ingest_runtime_timing_batch(
        self,
        timing_by_gpu: Dict[int, Dict[str, float]],
        ema: float = 0.4,
    ):
        """
        Batch ingestion helper.

        Example input:
            {
              0: {"compute_time": 0.120, "comm_time": 0.030},
              1: {"compute_time": 0.145, "comm_time": 0.035, "baseline_compute_time": 0.110}
            }
        """
        for gpu_id, m in timing_by_gpu.items():
            self.ingest_runtime_timing(
                gpu_id=int(gpu_id),
                compute_time=float(m.get("compute_time", 0.0)),
                comm_time=float(m.get("comm_time", 0.0)),
                baseline_compute_time=m.get("baseline_compute_time"),
                baseline_comm_time=m.get("baseline_comm_time"),
                ema=ema,
            )
        self._try_freeze_phase0_baseline()


def monitor_and_replan_with_mathematical_model(total_epochs: int = 1, steps_per_epoch: int = 1000):
    """
    수학적 모델을 사용한 모니터링 및 재분할
    기존 monitor_and_replan() 함수를 대체
    """
    optimizer = MathematicalFailoverOptimizer(total_epochs, steps_per_epoch)
    
    logger.info("🚀 Starting Mathematical Model Based Monitoring...")
    
    step_id = 0
    
    try:
        while step_id < steps_per_epoch * total_epochs:
            # 1. 훈련 진행상황 업데이트
            current_epoch = step_id // steps_per_epoch
            optimizer.update_training_progress(step_id, current_epoch)
            
            # 2. 각 GPU 성능 모니터링 (실제로는 GPU health monitor에서 호출)
            for gpu_id in range(4):  # 4개 GPU 가정
                # 실제 slowdown 측정 로직으로 교체
                simulated_slowdown = simulate_gpu_slowdown(gpu_id, step_id)

                # Runtime timing measurement hook (replace with real profiler metrics).
                compute_t, comm_t = simulate_gpu_timings(gpu_id, step_id)
                optimizer.ingest_runtime_timing(
                    gpu_id=gpu_id,
                    compute_time=compute_t,
                    comm_time=comm_t,
                    ema=0.4,
                )
                
                if simulated_slowdown > 1.05:  # 5% 이상 느려진 경우
                    # 3. 수학적 모델 기반 정책 결정
                    policy = optimizer.evaluate_slowdown_and_decide(gpu_id, simulated_slowdown)
                    
                    # 4. 정책 실행
                    optimizer.execute_policy(policy, gpu_id, simulated_slowdown)
            
            # 5. step 완료 시뮬레이션
            time.sleep(0.1)  # 실제로는 훈련 step 실행 시간
            step_id += 1
            
            # 주기적 상태 요약 출력
            if step_id % 100 == 0:
                summary = optimizer.get_current_performance_summary()
                logger.info(f"📊 Step {step_id} Summary: {summary}")
                
    except KeyboardInterrupt:
        logger.info("🛑 Monitoring stopped by user")
    
    final_summary = optimizer.get_current_performance_summary()
    logger.info(f"🏁 Final Summary: {json.dumps(final_summary, indent=2)}")


def benchmark_eta_overhead(
    num_trials: int = 100,
    slowdown: float = 1.3,
    target_gpu: int = 1,
) -> None:
    """ETA 재계산 오버헤드를 반복 측정해서 평균을 출력.

    CUDA_VISIBLE_DEVICES 등을 통해 물리 GPU (예: 1,4,5,6) 매핑은
    외부에서 설정한다고 가정하고, 여기서는 논리 GPU ID만 사용한다.
    """
    if not MATHEMATICAL_MODEL_AVAILABLE:
        logger.error("Mathematical model components are not available; cannot benchmark ETA.")
        return

    optimizer = MathematicalFailoverOptimizer(
        total_epochs=1,
        steps_per_epoch=200,
        baseline_warmup_steps=3,
    )

    # Slowdown 지속시간 조건 제거해서 바로 ETA 계산 경로를 타도록 설정
    optimizer.policy_selector.sustained_slowdown_duration = 0.0

    gpu_ids = list(range(4))  # 논리 GPU 0~3 (물리 GPU 매핑은 환경 변수로 조정)

    # Phase-0 baseline 수집 및 freeze (test_minimal_failover와 동일한 방식)
    for step in range(3):
        optimizer.update_training_progress(step_id=step, epoch=0)
        batch = {g: {"compute_time": 0.10, "comm_time": 0.02} for g in gpu_ids}
        optimizer.ingest_runtime_timing_batch(batch, ema=0.4)

    # warmup 경계 이후 한 번 더 호출해서 baseline freeze 유도
    optimizer.update_training_progress(step_id=optimizer.baseline_warmup_steps, epoch=0)
    if not optimizer._baseline_frozen:
        optimizer._try_freeze_phase0_baseline()

    # Failover gate(초기 step<5, K_rem<10)를 넘기기 위해 step을 충분히 진행시킨다.
    optimizer.update_training_progress(step_id=20, epoch=0)

    logical_target_gpu = target_gpu
    if logical_target_gpu not in gpu_ids:
        logical_target_gpu = gpu_ids[0]

    overheads_ms = []

    logger.info(
        f"🚀 Starting ETA overhead benchmark: trials={num_trials}, "
        f"slowdown={slowdown:.2f}, target_gpu={logical_target_gpu} (logical)"
    )

    for _ in range(num_trials):
        _policy = optimizer.evaluate_slowdown_and_decide(
            gpu_id=logical_target_gpu,
            current_slowdown=slowdown,
        )
        overhead = optimizer.policy_selector.last_eta_compute_ms
        if overhead is not None:
            overheads_ms.append(overhead)

    if not overheads_ms:
        logger.warning("No ETA overhead measurements collected.")
        return

    avg_ms = sum(overheads_ms) / len(overheads_ms)
    min_ms = min(overheads_ms)
    max_ms = max(overheads_ms)

    logger.info(
        f"📊 ETA computation overhead over {len(overheads_ms)} runs: "
        f"avg={avg_ms:.4f} ms, min={min_ms:.4f} ms, max={max_ms:.4f} ms"
    )

def simulate_gpu_slowdown(gpu_id: int, step_id: int) -> float:
    """GPU slowdown 시뮬레이션 (실제로는 GPU 모니터링에서 측정)"""
    # 특정 조건에서 slowdown 발생하도록 시뮬레이션
    if gpu_id == 1 and 200 <= step_id <= 400:  # GPU 1이 step 200-400에서 느려짐
        return 1.3  # 30% 느려짐
    elif gpu_id == 2 and step_id > 800:  # GPU 2가 후반부에 느려짐
        return 1.15  # 15% 느려짐
    return 1.0  # 정상


def simulate_gpu_timings(gpu_id: int, step_id: int) -> tuple[float, float]:
    """Synthetic compute/comm timings for demo of dynamic alpha/beta updates."""
    # Baseline-like timings.
    compute_t = 0.10
    comm_t = 0.03

    # Inject matching degradation into compute/comm channels.
    if gpu_id == 1 and 200 <= step_id <= 400:
        compute_t *= 1.30
        comm_t *= 1.15
    elif gpu_id == 2 and step_id > 800:
        compute_t *= 1.15
        comm_t *= 1.10

    return compute_t, comm_t


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mathematical failover optimizer utilities")
    parser.add_argument(
        "--mode",
        choices=["monitor", "eta_bench"],
        default="monitor",
        help="monitor: full monitoring loop, eta_bench: ETA overhead benchmark",
    )
    parser.add_argument("--eta-trials", type=int, default=100, help="ETA benchmark 반복 횟수")
    parser.add_argument("--eta-slowdown", type=float, default=1.3, help="ETA benchmark에서 사용할 slowdown 비율")
    parser.add_argument(
        "--eta-target-gpu",
        type=int,
        default=1,
        help="ETA benchmark용 논리 GPU ID (CUDA_VISIBLE_DEVICES로 물리 GPU 매핑)",
    )

    args = parser.parse_args()

    if args.mode == "monitor":
        monitor_and_replan_with_mathematical_model(total_epochs=1, steps_per_epoch=1000)
    else:
        benchmark_eta_overhead(
            num_trials=args.eta_trials,
            slowdown=args.eta_slowdown,
            target_gpu=args.eta_target_gpu,
        )
