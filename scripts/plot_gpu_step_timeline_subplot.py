#!/usr/bin/env python3
"""
Per-GPU Step Time Timeline Plotter (subplot version)
- 각 GPU별 subplot(4개)
- 원래 step time, moving average(10-step), 중앙값 라인
- 이벤트 마커(슬로우다운, 정책결정)는 해당 GPU subplot에만 표시

사용법 예시:
  python scripts/plot_gpu_step_timeline_subplot.py <experiment_dir>
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json

# 기존 plot_gpu_step_timeline.py의 데이터 파싱 함수들은 import 또는 복사 필요
# 여기서는 간단화를 위해 step_points_by_gpu, policies, injections만 가정

def moving_average(arr, window=10):
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(arr, np.ones(window)/window, mode='valid')

def median_line(arr, window=10):
    if len(arr) < window:
        return np.array(arr)
    return np.array([
        np.median(arr[max(0, i-window+1):i+1])
        for i in range(len(arr))
    ])

def load_step_points(profiling_dir):
    # gpu_task_summary_partition*.jsonl 파일에서 step별 time 추출
    step_points_by_gpu = defaultdict(list)
    for file in Path(profiling_dir).glob('gpu_task_summary_partition*.jsonl'):
        with open(file) as f:
            for line in f:
                rec = json.loads(line)
                gpu = int(rec['device'])
                step = int(rec['batch_id'])
                t = float(rec.get('exec_wall_ms', rec.get('time_ms', 0.0)))
                step_points_by_gpu[gpu].append((step, t))
    # step 오름차순 정렬
    for gpu in step_points_by_gpu:
        step_points_by_gpu[gpu].sort()
    return step_points_by_gpu

def plot_subplot(experiment_dir):
    profiling_dir = Path(experiment_dir) / 'profiling_logs'
    step_points_by_gpu = load_step_points(profiling_dir)
    gpu_ids = sorted(step_points_by_gpu.keys())[:4]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for idx, gpu in enumerate(gpu_ids):
        ax = axes[idx]
        steps, times = zip(*step_points_by_gpu[gpu])
        ax.plot(steps, times, label=f'GPU {gpu} Step Time', color='tab:blue', alpha=0.5)
        # Moving average
        ma = moving_average(times, window=10)
        ax.plot(steps[len(steps)-len(ma):], ma, label='Moving Avg (10)', color='tab:orange')
        # Median line
        med = median_line(times, window=10)
        ax.plot(steps, med, label='Median (10)', color='tab:green', linestyle='--')
        ax.set_ylabel('Step Time (ms)')
        ax.set_title(f'GPU {gpu}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Step')
    fig.suptitle('Per-GPU Step Time with Moving Average & Median')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(Path(experiment_dir) / 'per_gpu_step_time_subplot.png')
    print(f"Saved: {Path(experiment_dir) / 'per_gpu_step_time_subplot.png'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str, help='실험 결과 디렉토리')
    args = parser.parse_args()
    plot_subplot(args.experiment_dir)
