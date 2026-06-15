"""
Dynamic Alpha/Beta Estimator

Estimate per-GPU alpha_comp and beta_comm from measured runtime ratios,
then smooth updates with EMA for stability.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class AlphaBetaEstimate:
    gpu_id: int
    alpha_comp: float
    beta_comm: float


class DynamicAlphaBetaEstimator:
    """EMA-based alpha/beta updater from measured compute/comm ratios."""

    def __init__(self, initial_alpha: Optional[Dict[int, float]] = None, initial_beta: Optional[Dict[int, float]] = None):
        self.alpha_g: Dict[int, float] = {int(k): float(v) for k, v in (initial_alpha or {}).items()}
        self.beta_g: Dict[int, float] = {int(k): float(v) for k, v in (initial_beta or {}).items()}

    def update_from_ratios(
        self,
        gpu_id: int,
        compute_ratio: Optional[float] = None,
        comm_ratio: Optional[float] = None,
        ema: float = 0.4,
        min_ratio: float = 0.8,
        max_ratio: float = 10.0,
    ) -> AlphaBetaEstimate:
        """
        Update alpha/beta with measured ratios.

        Typical inputs:
        - compute_ratio = current_compute_time / baseline_compute_time
        - comm_ratio = current_comm_time / baseline_comm_time
        """
        gpu_id = int(gpu_id)
        old_alpha = self.alpha_g.get(gpu_id, 1.0)
        old_beta = self.beta_g.get(gpu_id, 1.0)

        if compute_ratio is not None:
            cr = float(compute_ratio)
            cr = max(min_ratio, min(max_ratio, cr))
            self.alpha_g[gpu_id] = ema * cr + (1.0 - ema) * old_alpha

        if comm_ratio is not None:
            br = float(comm_ratio)
            br = max(min_ratio, min(max_ratio, br))
            self.beta_g[gpu_id] = ema * br + (1.0 - ema) * old_beta

        return AlphaBetaEstimate(
            gpu_id=gpu_id,
            alpha_comp=self.alpha_g.get(gpu_id, old_alpha),
            beta_comm=self.beta_g.get(gpu_id, old_beta),
        )

    def get_coefficients(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        return dict(self.alpha_g), dict(self.beta_g)
