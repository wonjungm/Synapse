import os
import json
from statistics import mean
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
# user-provided path
RESULTS_DIR = '/acpl-ssd10/Synapse-private/benchmarks/soft_target/results/exp0_final'

THRESHOLD_SPIKE_MS = 1000.0

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def discover_experiments(results_root):
    if not os.path.exists(results_root):
        return []
    return sorted([d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))])


EXPS = discover_experiments(RESULTS_DIR)

results = {}
for exp in EXPS:
    d = os.path.join(RESULTS_DIR, exp)
    step_file = os.path.join(d, 'exp0_checkpoint_step_metrics.jsonl')
    save_file = os.path.join(d, 'exp0_checkpoint_save_events.jsonl')
    step_events = read_jsonl(step_file)
    save_events = read_jsonl(save_file)

    # Build step list
    step_list = []
    for e in step_events:
        if e.get('event_type') != 'step_timing':
            continue
        sid = e.get('step_id')
        t = e.get('step_time_ms')
        if isinstance(t, (int, float)):
            step_list.append({'step_id': sid, 'step_time_ms': t, 'is_checkpoint': False})

    # Determine checkpoint step ids from save and spike events
    checkpoint_step_ids = set()
    for e in save_events:
        # pick common numeric fields
        for key in ('post_step_id', 'batch_count', 'step_id', 'save_step'):
            v = e.get(key)
            if isinstance(v, int):
                checkpoint_step_ids.add(v)
    for e in save_events:
        if e.get('event_type') == 'checkpoint_spike_observed':
            v = e.get('post_step_id')
            if isinstance(v, int):
                checkpoint_step_ids.add(v)

    # mark checkpoint steps in list
    for s in step_list:
        if isinstance(s.get('step_id'), int) and s['step_id'] in checkpoint_step_ids:
            s['is_checkpoint'] = True

    step_times = [s['step_time_ms'] for s in step_list]
    step_times_nonzero = [t for t in step_times if t and t > 0]
    step_times_excluding_checkpoints = [s['step_time_ms'] for s in step_list if not s['is_checkpoint'] and s['step_time_ms'] and s['step_time_ms'] > 0]

    avg_excluding = mean(step_times_excluding_checkpoints) if step_times_excluding_checkpoints else None
    avg_including = mean(step_times_nonzero) if step_times_nonzero else None

    c_saves = []
    spike_summary = []
    file_sizes_reported = set()
    for e in save_events:
        if e.get('event_type') == 'checkpoint_save':
            c_saves.append({'save_duration_sec': e.get('save_duration_sec'), 'file_size_bytes': e.get('file_size_bytes'), 'save_id': e.get('save_id'), 'batch_count': e.get('batch_count')})
            if e.get('file_size_bytes'):
                file_sizes_reported.add(e.get('file_size_bytes'))
        if e.get('event_type') == 'checkpoint_spike_observed':
            spike_summary.append({'save_id': e.get('save_id'), 'post_step_id': e.get('post_step_id'), 'post_step_time_ms': e.get('post_step_time_ms'), 'delta_ms': e.get('delta_ms'), 'ratio_post_over_pre': e.get('ratio_post_over_pre')})

    # Verify on-disk checkpoint file
    on_disk_size = None
    ckpt_path = os.path.join(d, 'healthy_checkpoint_latest.pth')
    if os.path.exists(ckpt_path):
        try:
            on_disk_size = os.path.getsize(ckpt_path)
        except Exception:
            on_disk_size = None

    results[exp] = {
        'avg_excluding_checkpoint_ms': avg_excluding,
        'avg_including_ms': avg_including,
        'num_steps_total': len(step_times),
        'num_checkpoint_steps': len([s for s in step_list if s['is_checkpoint']]),
        'c_saves': c_saves,
        'spike_summary': spike_summary,
        'reported_file_sizes': list(file_sizes_reported),
        'on_disk_checkpoint_bytes': on_disk_size,
        'step_list': step_list,
    }

# Print concise report
for exp, r in results.items():
    print('==', exp)
    print('Avg step (ms) excluding checkpoints:', r['avg_excluding_checkpoint_ms'])
    print('Avg step (ms) including checkpoints:', r['avg_including_ms'])
    print('Steps total / checkpoint-steps:', r['num_steps_total'], '/', r['num_checkpoint_steps'])
    print('C_save entries (count):', len(r['c_saves']))
    if r['c_saves']:
        print('C_save durations (s):', [round(x['save_duration_sec'], 4) if x['save_duration_sec'] is not None else None for x in r['c_saves']])
        print('Reported file_size_bytes (unique):', r['reported_file_sizes'])
    if r['spike_summary']:
        print('Spike summary (first 5):')
        for s in r['spike_summary'][:5]:
            print('  ', s)
    print('On-disk healthy_checkpoint_latest.pth size:', r['on_disk_checkpoint_bytes'])
    print()

# Combined CSV summary (one row per experiment) with four main values plus extras
CSV_OUT = os.path.join(os.path.dirname(__file__), 'checkpoint_summary.csv')
with open(CSV_OUT, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['experiment', 'avg_excluding_checkpoint_ms', 'avg_including_ms', 'mean_C_save_s', 'num_C_saves', 'reported_file_size_bytes', 'on_disk_checkpoint_bytes', 'num_steps_total', 'num_checkpoint_steps'])
    for exp, r in results.items():
        mean_c = None
        if r['c_saves']:
            vals = [x['save_duration_sec'] for x in r['c_saves'] if isinstance(x.get('save_duration_sec'), (int, float))]
            mean_c = mean(vals) if vals else None
        writer.writerow([exp, r['avg_excluding_checkpoint_ms'], r['avg_including_ms'], mean_c, len(r['c_saves']), r['reported_file_sizes'][0] if r['reported_file_sizes'] else None, r['on_disk_checkpoint_bytes'], r['num_steps_total'], r['num_checkpoint_steps']])

# Per-experiment detailed CSVs (step times + checkpoint flag)
for exp, r in results.items():
    out_detail = os.path.join(os.path.dirname(__file__), f'{exp}_steps.csv')
    with open(out_detail, 'w', newline='') as df:
        w = csv.writer(df)
        w.writerow(['step_id', 'step_time_ms', 'is_checkpoint'])
        for s in r['step_list']:
            w.writerow([s.get('step_id'), s.get('step_time_ms'), s.get('is_checkpoint')])

# Visualization: two panels
png_out = os.path.join(os.path.dirname(__file__), 'checkpoint_summary.png')
exps = list(results.keys())
avg_excl = [results[e]['avg_excluding_checkpoint_ms'] or 0 for e in exps]
avg_incl = [results[e]['avg_including_ms'] or 0 for e in exps]
mean_c_list = []
for e in exps:
    rs = results[e]
    if rs['c_saves']:
        vals = [x['save_duration_sec'] for x in rs['c_saves'] if isinstance(x.get('save_duration_sec'), (int, float))]
        mean_c_list.append(mean(vals) if vals else 0)
    else:
        mean_c_list.append(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ind = range(len(exps))
width = 0.35
axes[0].bar([i - width/2 for i in ind], avg_excl, width=width, label='avg_excluding_checkpoint_ms')
axes[0].bar([i + width/2 for i in ind], avg_incl, width=width, label='avg_including_ms')
axes[0].set_xticks(ind)
axes[0].set_xticklabels(exps, rotation=45, ha='right')
axes[0].set_ylabel('ms')
axes[0].set_title('Average step time (excluding vs including checkpoints)')
axes[0].legend()

axes[1].bar(ind, mean_c_list, color='C2')
axes[1].set_xticks(ind)
axes[1].set_xticklabels(exps, rotation=45, ha='right')
axes[1].set_ylabel('seconds')
axes[1].set_title('Mean checkpoint save duration (C_save)')

plt.tight_layout()
plt.savefig(png_out)
print('Wrote CSV:', CSV_OUT)
print('Wrote plot PNG:', png_out)
