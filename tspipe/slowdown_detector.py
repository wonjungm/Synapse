import time
import os
import numpy as np
from collections import deque
from typing import Any, Dict, Optional
import logging


class SlowdownDetector:
    """Stage time 기반 wall-clock slowdown 탐지 및 sustained trigger 관리."""

    def __init__(
        self,
        baseline_window: int = 10,
        detection_window: int = 5,
        slowdown_threshold: float = 1.10,
        inject_scenario: str = "",
        baseline_skip_steps: Optional[int] = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.SlowdownDetector")

        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.slowdown_threshold = slowdown_threshold
        self.inject_scenario = inject_scenario.strip()
        if baseline_skip_steps is None:
            baseline_skip_steps = int(
                os.environ.get("FAILOVER_BASELINE_SKIP_STEPS", "20").strip() or 20
            )
        self.baseline_skip_steps = max(0, int(baseline_skip_steps))

        self.stage_times = deque(maxlen=max(baseline_window, detection_window))
        self._baseline_candidates = deque(maxlen=max(1, baseline_window))

        self.baseline_stage_time: Optional[float] = None
        self.baseline_std: Optional[float] = None

        self.batch_count = 0
        self.slowdown_detected_at_step = None
        self.slowdown_detected_at_global_step = None
        self.last_global_step: Optional[int] = None

        # wall-clock sustained trigger state
        self.wallclock_slowdown_started_at: Optional[float] = None
        self.wallclock_sustained_duration_sec: float = 0.0

        self.logger.info("SlowdownDetector initialized")

    def record_stage_time(
        self,
        stage_time_ms: float,
        timestamp_sec: Optional[float] = None,
        global_step: Optional[int] = None,
    ):
        
        # 매 step마다 wall-clock elapsed time(ms)를 기록
        
        self.stage_times.append(float(stage_time_ms))
        self.batch_count += 1
        self.last_global_step = None if global_step is None else int(global_step)

        now = float(timestamp_sec) if timestamp_sec is not None else time.time()

        # baseline 확정: skip 초기 warmup 후, 다음 baseline_window samples의 median 사용
        if self.baseline_stage_time is None:
            if self.batch_count > self.baseline_skip_steps:
                self._baseline_candidates.append(float(stage_time_ms))

            if len(self._baseline_candidates) >= self.baseline_window:
                samples = list(self._baseline_candidates)
                self.baseline_stage_time = float(np.median(samples))
                self.baseline_std = float(np.std(samples))
                self.logger.info(
                    f"📊 Baseline set after skip={self.baseline_skip_steps}: "
                    f"{self.baseline_stage_time:.2f}ms (±{self.baseline_std:.2f}ms, "
                    f"window={self.baseline_window}, method=median)"
                )

        if self.baseline_stage_time is not None:
            self._update_wallclock_trigger_state(now)

    def _update_wallclock_trigger_state(self, now: float):
        slowdown = self.get_slowdown_ratio()

        if slowdown > self.slowdown_threshold:
            if self.wallclock_slowdown_started_at is None:
                self.wallclock_slowdown_started_at = now
                self.wallclock_sustained_duration_sec = 0.0
            else:
                self.wallclock_sustained_duration_sec = max(
                    0.0, now - self.wallclock_slowdown_started_at
                )

            if self.slowdown_detected_at_step is None:
                self.slowdown_detected_at_step = self.batch_count
                self.slowdown_detected_at_global_step = self.last_global_step
                if self.last_global_step is not None:
                    self.logger.warning(
                        "⚠️ Wall-clock slowdown detected "
                        f"(local_step={self.batch_count}, global_step={self.last_global_step}): "
                        f"{slowdown:.3f}x (threshold: {self.slowdown_threshold:.2f})"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ Wall-clock slowdown detected at step {self.batch_count}: "
                        f"{slowdown:.3f}x (threshold: {self.slowdown_threshold:.2f})"
                    )
        else:
            self.wallclock_slowdown_started_at = None
            self.wallclock_sustained_duration_sec = 0.0
            self.slowdown_detected_at_step = None
            self.slowdown_detected_at_global_step = None

    def get_slowdown_ratio(self) -> float:
        if self.baseline_stage_time is None:
            return 1.0

        if len(self.stage_times) >= self.detection_window:
            recent = list(self.stage_times)[-self.detection_window:]
            avg_recent = float(np.mean(recent))
            slowdown = avg_recent / self.baseline_stage_time
            return slowdown

        return 1.0

    def is_slowdown_detected(self, threshold: Optional[float] = None) -> bool:
        threshold = self.slowdown_threshold if threshold is None else float(threshold)
        return self.get_slowdown_ratio() > threshold

    def get_trigger_state(self, sustain_sec: float) -> Dict:
        slowdown = self.get_slowdown_ratio()
        triggered = (
            slowdown > self.slowdown_threshold
            and self.wallclock_sustained_duration_sec >= float(sustain_sec)
        )
        return {
            "current_slowdown_ratio": slowdown,
            "sustained_duration_sec": self.wallclock_sustained_duration_sec,
            "triggered": triggered,
            "threshold": self.slowdown_threshold,
            "sustain_sec_required": float(sustain_sec),
            "baseline_stage_time_ms": self.baseline_stage_time,
            "baseline_std_ms": self.baseline_std,
            "batch_count": self.batch_count,
            "global_step": self.last_global_step,
            "slowdown_detected_at_step": self.slowdown_detected_at_step,
            "slowdown_detected_at_global_step": self.slowdown_detected_at_global_step,
        }

    def get_statistics(self) -> Dict:
        if self.baseline_stage_time is None:
            samples_needed = self.baseline_skip_steps + self.baseline_window - self.batch_count
            return {
                "status": "baseline_not_set",
                "batch_count": self.batch_count,
                "baseline_required": max(0, samples_needed),
                "baseline_skip_steps": self.baseline_skip_steps,
            }

        slowdown = self.get_slowdown_ratio()
        recent = list(self.stage_times)[-self.detection_window:] if self.stage_times else []

        return {
            "status": "normal" if not self.is_slowdown_detected() else "slowdown_detected",
            "batch_count": self.batch_count,
            "baseline_stage_time_ms": self.baseline_stage_time,
            "baseline_std_ms": self.baseline_std,
            "baseline_skip_steps": self.baseline_skip_steps,
            "current_slowdown_ratio": slowdown,
            "recent_avg_ms": float(np.mean(recent)) if recent else None,
            "recent_min_ms": float(np.min(recent)) if recent else None,
            "recent_max_ms": float(np.max(recent)) if recent else None,
            "wallclock_sustained_duration_sec": self.wallclock_sustained_duration_sec,
            "slowdown_detected_at_step": self.slowdown_detected_at_step,
            "slowdown_detected_at_global_step": self.slowdown_detected_at_global_step,
            "global_step": self.last_global_step,
        }


    def export_restart_state(self) -> Dict[str, Any]:
        """Serialize detector state for a failover restart."""
        return {
            "baseline_stage_time_ms": (
                None if self.baseline_stage_time is None else float(self.baseline_stage_time)
            ),
            "baseline_std_ms": None if self.baseline_std is None else float(self.baseline_std),
            "baseline_skip_steps": int(self.baseline_skip_steps),
            "batch_count": int(self.batch_count),
            "last_global_step": (
                None if self.last_global_step is None else int(self.last_global_step)
            ),
            "stage_times_ms": [float(v) for v in self.stage_times],
            "baseline_candidate_times_ms": [float(v) for v in self._baseline_candidates],
            "wallclock_sustained_duration_sec": float(self.wallclock_sustained_duration_sec),
            "slowdown_detected_at_step": (
                None if self.slowdown_detected_at_step is None else int(self.slowdown_detected_at_step)
            ),
            "slowdown_detected_at_global_step": (
                None if self.slowdown_detected_at_global_step is None else int(self.slowdown_detected_at_global_step)
            ),
        }

    def restore_restart_state(
        self,
        state: Optional[Dict[str, Any]],
        clear_recent_window: bool = True,
        clear_trigger_state: bool = True,
    ) -> bool:
        """Restore detector state while avoiding accidental post-restart re-baselining."""
        if not isinstance(state, dict):
            return False

        baseline = state.get("baseline_stage_time_ms")
        baseline_std = state.get("baseline_std_ms")
        self.baseline_stage_time = None if baseline is None else float(baseline)
        self.baseline_std = None if baseline_std is None else float(baseline_std)
        self.baseline_skip_steps = int(state.get("baseline_skip_steps", self.baseline_skip_steps) or 0)
        self.batch_count = int(state.get("batch_count", 0) or 0)

        last_global_step = state.get("last_global_step")
        self.last_global_step = None if last_global_step is None else int(last_global_step)

        self.stage_times.clear()
        self._baseline_candidates.clear()
        if not clear_recent_window:
            for value in state.get("stage_times_ms", [])[-self.stage_times.maxlen:]:
                self.stage_times.append(float(value))
        for value in state.get("baseline_candidate_times_ms", [])[-self._baseline_candidates.maxlen:]:
            self._baseline_candidates.append(float(value))

        if clear_trigger_state:
            self.wallclock_slowdown_started_at = None
            self.wallclock_sustained_duration_sec = 0.0
            self.slowdown_detected_at_step = None
            self.slowdown_detected_at_global_step = None
        else:
            sustained = float(state.get("wallclock_sustained_duration_sec", 0.0) or 0.0)
            self.wallclock_sustained_duration_sec = sustained
            self.wallclock_slowdown_started_at = None if sustained <= 0 else time.time() - sustained
            detected_at_step = state.get("slowdown_detected_at_step")
            self.slowdown_detected_at_step = None if detected_at_step is None else int(detected_at_step)
            detected_at_global_step = state.get("slowdown_detected_at_global_step")
            self.slowdown_detected_at_global_step = (
                None if detected_at_global_step is None else int(detected_at_global_step)
            )

        if self.baseline_stage_time is not None:
            self.logger.info(
                "✅ Restored slowdown detector baseline from checkpoint: "
                f"{self.baseline_stage_time:.2f}ms"
            )
            if clear_recent_window:
                self.logger.info(
                    "↪️ Cleared detector recent window after restart; new decisions will use "
                    "fresh post-restart samples against the preserved baseline"
                )
        return self.baseline_stage_time is not None

    def set_baseline(
        self,
        baseline_stage_time_ms: float,
        baseline_std_ms: float = 0.0,
        clear_recent_window: bool = True,
        clear_trigger_state: bool = True,
    ) -> None:
        """Override the detector baseline without forcing a new measured warmup window."""
        self.baseline_stage_time = float(baseline_stage_time_ms)
        self.baseline_std = float(baseline_std_ms)
        if clear_recent_window:
            self.stage_times.clear()
            self._baseline_candidates.clear()
        if clear_trigger_state:
            self.wallclock_slowdown_started_at = None
            self.wallclock_sustained_duration_sec = 0.0
            self.slowdown_detected_at_step = None
            self.slowdown_detected_at_global_step = None
        self.logger.info(
            "🧭 Slowdown detector baseline overridden: "
            f"{self.baseline_stage_time:.2f}ms (std={self.baseline_std:.2f}ms)"
        )
