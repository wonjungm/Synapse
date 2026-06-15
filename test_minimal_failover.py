#!/usr/bin/env python3
"""
Planner regression scenario test:
Phase-0 complete -> slight slowdown (KEEP) -> sustained slowdown (REPLAN) -> severe slowdown (DEGRADE).
"""

from benchmarks.soft_target.planner.mathematical_optimizer import MathematicalFailoverOptimizer
from benchmarks.soft_target.planner.stage_time_predictor import PartitionConfig


def _feed_baseline(optimizer: MathematicalFailoverOptimizer, steps: int, gpus: list[int]) -> None:
    for step in range(steps):
        optimizer.update_training_progress(step_id=step, epoch=0)
        batch = {gpu: {"compute_time": 0.10, "comm_time": 0.02} for gpu in gpus}
        optimizer.ingest_runtime_timing_batch(batch, ema=0.4)


def run_regression_scenario() -> None:
    optimizer = MathematicalFailoverOptimizer(
        total_epochs=1,
        steps_per_epoch=200,
        baseline_warmup_steps=3,
    )

    # Make this deterministic for a fast test (no 30s wait for sustained check).
    optimizer.policy_selector.sustained_slowdown_duration = 0.0

    gpu_ids = [0, 1, 2, 3]

    print("[1] Phase-0 baseline collection and freeze")
    _feed_baseline(optimizer, steps=3, gpus=gpu_ids)
    optimizer.update_training_progress(step_id=3, epoch=0)

    if not optimizer._baseline_frozen:
        # Force a freeze check once after warmup boundary.
        optimizer._try_freeze_phase0_baseline()

    # Clear early-training failover gate (current_step must be >= 5).
    optimizer.update_training_progress(step_id=20, epoch=0)

    # Intentionally imbalanced layout so GPU 1 slowdown has visible ETA impact.
    optimizer.current_partition = PartitionConfig(
        snet_partition=[2, 18, 3, 3],
        tnet_partition=[2, 12, 2, 2],
        gpu_assignment=[0, 1, 2, 3],
    )

    print(f"  baseline_frozen={optimizer._baseline_frozen}")
    print(f"  initial_partition={optimizer.current_partition}")

    print("[2] Slight slowdown -> KEEP expected")
    # Keep alpha/beta close to baseline and use mild slowdown.
    optimizer.ingest_runtime_timing_batch({1: {"compute_time": 0.102, "comm_time": 0.0205}}, ema=0.1)
    policy_keep = optimizer.evaluate_slowdown_and_decide(gpu_id=1, current_slowdown=1.03)
    print(f"  policy={policy_keep}")
    assert policy_keep == "KEEP", f"Expected KEEP, got {policy_keep}"

    print("[3] Sustained slowdown -> REPLAN expected + partition update")
    before_replan = optimizer.current_partition
    optimizer.ingest_runtime_timing_batch({1: {"compute_time": 0.14, "comm_time": 0.028}}, ema=0.3)
    policy_replan = optimizer.evaluate_slowdown_and_decide(gpu_id=1, current_slowdown=1.30)
    print(f"  policy={policy_replan}")
    assert policy_replan == "REPLAN", f"Expected REPLAN, got {policy_replan}"
    optimizer.execute_policy(policy_replan, gpu_id=1, current_slowdown=1.30)
    after_replan = optimizer.current_partition
    print(f"  partition_before={before_replan}")
    print(f"  partition_after ={after_replan}")
    assert before_replan != after_replan, "Expected partition change after REPLAN"

    print("[4] GPU failure -> DEGRADE expected + failed GPU exclusion")
    optimizer.ingest_runtime_timing_batch({1: {"compute_time": 5.0, "comm_time": 1.0}}, ema=0.9)
    policy_degrade = optimizer.evaluate_slowdown_and_decide(gpu_id=1, current_slowdown=3.00, failed_gpus=[1])
    print(f"  policy={policy_degrade}")
    assert policy_degrade == "DEGRADE", f"Expected DEGRADE, got {policy_degrade}"
    optimizer.execute_policy(policy_degrade, gpu_id=1, current_slowdown=3.00)

    print(f"  final_partition={optimizer.current_partition}")
    assert 1 not in optimizer.current_partition.gpu_assignment, "Expected degraded partition to exclude GPU 1"

    print("\nPASS: KEEP -> REPLAN -> DEGRADE scenario verified")


if __name__ == "__main__":
    run_regression_scenario()
