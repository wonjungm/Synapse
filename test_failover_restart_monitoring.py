#!/usr/bin/env python3
"""Regression checks for post-restart failover monitoring state."""

import csv
import sys
import types
from pathlib import Path


class _FakeSeries(list):
    def tolist(self):
        return list(self)


class _FakeDataFrame:
    def __init__(self, columns):
        self._columns = columns

    def __getitem__(self, key):
        return _FakeSeries(self._columns[key])

    def __len__(self):
        if not self._columns:
            return 0
        first_key = next(iter(self._columns))
        return len(self._columns[first_key])


def _fake_read_csv(path):
    with open(Path(path), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        columns = {name: [] for name in fieldnames}
        for row in reader:
            for name in fieldnames:
                value = row.get(name, "")
                try:
                    columns[name].append(float(value))
                except (TypeError, ValueError):
                    columns[name].append(value)
    return _FakeDataFrame(columns)


sys.modules.setdefault("pandas", types.SimpleNamespace(read_csv=_fake_read_csv))

from benchmarks.soft_target.planner.mathematical_optimizer import MathematicalFailoverOptimizer
from benchmarks.soft_target.planner.stage_time_predictor import PartitionConfig
from tspipe.slowdown_detector import SlowdownDetector


def _feed_baseline(optimizer: MathematicalFailoverOptimizer, steps: int, gpus: list[int]) -> None:
    for step in range(steps):
        optimizer.update_training_progress(step_id=step, epoch=0)
        batch = {gpu: {"compute_time": 0.10, "comm_time": 0.02} for gpu in gpus}
        optimizer.ingest_runtime_timing_batch(batch, ema=0.4)


def _verify_skip_then_median_baseline() -> None:
    detector = SlowdownDetector(
        baseline_window=4,
        detection_window=3,
        slowdown_threshold=1.10,
        baseline_skip_steps=3,
    )
    samples = [1000.0, 5000.0, 900.0, 100.0, 110.0, 120.0, 130.0]
    for step, value in enumerate(samples, start=1):
        detector.record_stage_time(value, global_step=step)

    assert detector.baseline_stage_time is not None
    assert abs(detector.baseline_stage_time - 115.0) < 1e-6
    assert detector.baseline_skip_steps == 3


def run_regression() -> None:
    _verify_skip_then_median_baseline()

    optimizer = MathematicalFailoverOptimizer(total_epochs=1, steps_per_epoch=200, baseline_warmup_steps=3)
    gpus = [0, 1, 2, 3]
    _feed_baseline(optimizer, steps=3, gpus=gpus)
    optimizer.update_training_progress(step_id=20, epoch=0)
    if not optimizer._baseline_frozen:
        optimizer._try_freeze_phase0_baseline()

    previous_partition = PartitionConfig(
        snet_partition=[14, 14, 14, 14],
        tnet_partition=[7, 7, 7, 7],
        gpu_assignment=[0, 1, 2, 3],
    )
    new_partition = PartitionConfig(
        snet_partition=[8, 16, 16, 16],
        tnet_partition=[4, 8, 8, 8],
        gpu_assignment=[0, 1, 2, 3],
    )

    detector = SlowdownDetector(baseline_window=4, detection_window=3, slowdown_threshold=1.10)
    detector.set_baseline(1943.07, 422.11)
    detector.batch_count = 100
    detector.last_global_step = 100
    detector.record_stage_time(2300.0, global_step=101)
    detector.record_stage_time(2350.0, global_step=102)
    detector.record_stage_time(2400.0, global_step=103)

    detector_state = detector.export_restart_state()
    optimizer_state = optimizer.export_restart_state()

    previous_nominal = optimizer.estimate_partition_nominal_step_time(previous_partition)
    new_nominal = optimizer.estimate_partition_nominal_step_time(new_partition)
    assert previous_nominal > 0.0 and new_nominal > 0.0

    scaled_baseline = detector_state["baseline_stage_time_ms"] * (new_nominal / previous_nominal)
    assert scaled_baseline > 0.0
    assert abs(scaled_baseline - detector_state["baseline_stage_time_ms"]) > 1e-6

    restored_detector = SlowdownDetector(baseline_window=4, detection_window=3, slowdown_threshold=1.10)
    restored_optimizer = MathematicalFailoverOptimizer(total_epochs=1, steps_per_epoch=200, baseline_warmup_steps=3)
    restored_optimizer.current_partition = new_partition
    restored_optimizer.restore_restart_state(optimizer_state)

    restored_detector.restore_restart_state(
        detector_state,
        clear_recent_window=True,
        clear_trigger_state=True,
    )
    restored_detector.set_baseline(
        baseline_stage_time_ms=scaled_baseline,
        baseline_std_ms=detector_state["baseline_std_ms"],
        clear_recent_window=True,
        clear_trigger_state=True,
    )

    assert restored_optimizer._baseline_frozen is True
    assert restored_detector.baseline_stage_time is not None
    assert abs(restored_detector.baseline_stage_time - scaled_baseline) < 1e-6
    assert len(restored_detector.stage_times) == 0

    restored_detector.record_stage_time(restored_detector.baseline_stage_time * 1.25, global_step=104)
    restored_detector.record_stage_time(restored_detector.baseline_stage_time * 1.28, global_step=105)
    restored_detector.record_stage_time(restored_detector.baseline_stage_time * 1.30, global_step=106)
    assert restored_detector.get_slowdown_ratio() > 1.10

    print("PASS: restart-state restoration preserves optimizer baseline and partition-adjusted detector baseline")


if __name__ == "__main__":
    run_regression()
