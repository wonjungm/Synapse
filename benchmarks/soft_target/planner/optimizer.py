import pandas as pd
import numpy as np
import sys
import torch
import time


tnet_df = pd.read_csv('./profile/tnet.csv') # layer-wise
snet_df = pd.read_csv('./profile/snet.csv')

tnet_num_layers = len(tnet_df)
snet_num_layers = len(snet_df)

tnet_forward_times_ms = tnet_df['forward_time_ms'].tolist()
tnet_param_sizes_kb = tnet_df['parameter_size_kb'].tolist()
tnet_input_activation_sizes_kb = tnet_df['input_activation_size_kb'].tolist()
tnet_output_activation_sizes_kb = tnet_df['output_activation_size_kb'].tolist() # not used
tnet_accum_activation_sizes_kb = tnet_df['accum_activation_size_kb'].tolist()

snet_forward_times_ms = snet_df['forward_time_ms'].tolist()
snet_backward_times_ms = snet_df['backward_time_ms'].tolist()
snet_param_sizes_kb = snet_df['parameter_size_kb'].tolist()
snet_input_activation_sizes_kb = snet_df['input_activation_size_kb'].tolist()
snet_output_activation_sizes_kb = snet_df['output_activation_size_kb'].tolist()
snet_accum_activation_sizes_kb = snet_df['accum_activation_size_kb'].tolist()


# =======
bandwidth_gbps = 8  # GB/s
bandwidth_kbps = bandwidth_gbps * 1024 * 1024  # GB/s → KB/s
device = torch.device("cuda:0")  # 첫 번째 GPU
total_memory = torch.cuda.get_device_properties(device).total_memory  # bytes   # device memory constraint (per stage)
total_memory_kb = total_memory / 1024  # Convert to KB
print(f"Total memory available on device: {total_memory_kb:.2f} KB")
num_stages = 4

def get_snet_stage_time(layer_i, layer_j):
    fwd_time = np.sum(snet_forward_times_ms[layer_i:layer_j+1]) * 1e-3  # ms -> s
    bwd_time = np.sum(snet_backward_times_ms[layer_i:layer_j+1]) * 1e-3  # ms -> s
    
    last_layer = snet_num_layers - 1
    recv_act_time = 0
    recv_grad_time = 0
    
    if layer_i != 0:
        recv_act_size = snet_input_activation_sizes_kb[layer_i]
        recv_act_time = recv_act_size / bandwidth_kbps  # time to receive activations in seconds
    if layer_j != last_layer:
        recv_grad_size = snet_output_activation_sizes_kb[layer_j]
        recv_grad_time = recv_grad_size / bandwidth_kbps  # time to receive gradients in seconds

    snet_total_time = recv_act_time + fwd_time + recv_grad_time + bwd_time
    return snet_total_time

def get_tnet_stage_time(layer_i, layer_j):
    fwd_time = np.sum(tnet_forward_times_ms[layer_i:layer_j+1]) * 1e-3  # ms -> s
    recv_act_time = 0
    
    if layer_i != 0:
        recv_act_size = tnet_input_activation_sizes_kb[layer_i]
        recv_act_time = recv_act_size / bandwidth_kbps  # time to receive activations in seconds

    tnet_total_time = recv_act_time + fwd_time
    return tnet_total_time

snet_dp = np.full((snet_num_layers + 1, num_stages + 1), np.inf)
tnet_dp = np.full((tnet_num_layers + 1, num_stages + 1), np.inf)
snet_partition = np.full((snet_num_layers + 1, num_stages + 1), -1)
tnet_partition = np.full((tnet_num_layers + 1, num_stages + 1), -1)

snet_dp[0][0] = 0  # base case
tnet_dp[0][0] = 0  # base case


t0 = time.perf_counter()
for i in range(1, snet_num_layers+1):
    for s in range(1, num_stages+1):
        for j in range(s-1, i):
            snet_time = get_snet_stage_time(j, i-1)
            candidate = max(snet_dp[j][s-1], snet_time)
            if candidate < snet_dp[i][s]:
                snet_dp[i][s] = candidate
                snet_partition[i][s] = j

for i in range(1, tnet_num_layers+1):
    for s in range(1, num_stages+1):
        for j in range(s-1, i):
            tnet_time = get_tnet_stage_time(j, i-1)
            candidate = max(tnet_dp[j][s-1], tnet_time)
            if candidate < tnet_dp[i][s]:
                tnet_dp[i][s] = candidate
                tnet_partition[i][s] = j
                
dp_duration = (time.perf_counter() - t0) * 1000
print(f"\n[Info] DP duration: {dp_duration:.3f} ms\n")

def reconstruct_partition(partition_table, num_layers, num_stages):
    lengths = []
    curr_layer = num_layers
    for s in reversed(range(1, num_stages + 1)):
        prev_layer = partition_table[curr_layer][s]
        lengths.append(curr_layer - prev_layer)
        curr_layer = prev_layer
    return list(reversed(lengths))

snet_split = reconstruct_partition(snet_partition, snet_num_layers, num_stages)
tnet_split = reconstruct_partition(tnet_partition, tnet_num_layers, num_stages)

print("SNet partition:", snet_split)
print("TNet partition:", tnet_split)




# ======== Memory constraints check with recompute ========

def get_snet_stage_memory(layer_i, layer_j):
    act_mem = np.sum(snet_accum_activation_sizes_kb[layer_i:layer_j+1])
    param_mem = np.sum(snet_param_sizes_kb[layer_i:layer_j+1])
    return act_mem + param_mem

def get_tnet_stage_memory(layer_i, layer_j):
    act_mem = np.sum(tnet_accum_activation_sizes_kb[layer_i:layer_j+1])
    param_mem = np.sum(tnet_param_sizes_kb[layer_i:layer_j+1])
    return act_mem + param_mem

def partition_to_indices(partition_lengths):
    indices = []
    start = 0
    for length in partition_lengths:
        end = start + length
        indices.append((start, end))
        start = end
    return indices

def check_memory_constraints_with_recompute(partition_lengths, get_stage_memory, input_activation_kb_list, param_kb_list, total_memory_kb):
    indices = partition_to_indices(partition_lengths)
    result = []

    for i, (start, end) in enumerate(indices):
        original_mem = get_stage_memory(start, end - 1)
        param_mem = np.sum(param_kb_list[start:end])
        input_activation = input_activation_kb_list[start]
        recomputed_mem = param_mem + input_activation

        if original_mem <= total_memory_kb:
            result.append(("OK", original_mem))
        elif recomputed_mem <= total_memory_kb:
            result.append(("RECOMPUTE", recomputed_mem))
        else:
            result.append(("FAIL", original_mem))
    return result


snet_mem_status = check_memory_constraints_with_recompute(
    snet_split,
    get_snet_stage_memory,
    snet_input_activation_sizes_kb,
    snet_param_sizes_kb,
    total_memory_kb
)

tnet_mem_status = check_memory_constraints_with_recompute(
    tnet_split,
    get_tnet_stage_memory,
    tnet_input_activation_sizes_kb,
    tnet_param_sizes_kb,
    total_memory_kb
)

for i, (snet, tnet) in enumerate(zip(snet_mem_status, tnet_mem_status)):
    print(f"Stage {i}: SNet = {snet}, TNet = {tnet}")
    
    
    
# ----------------------------------------------------------------------
#  Utility: Stage-Time Imbalance Ratio
# ----------------------------------------------------------------------
def imbalance_ratio(partition_lengths, get_stage_time):
    times = []
    start = 0
    for length in partition_lengths:
        end = start + length - 1         # inclusive index
        times.append(get_stage_time(start, end))
        start = end + 1

    worst  = max(times)
    mean   = sum(times) / len(times)
    ratio  = worst / mean if mean > 0 else float('inf')
    return ratio, times


optimed_snet_ratio, optimed_snet_stage_times = imbalance_ratio(snet_split, get_snet_stage_time)
default_snet_ratio, default_snet_stage_times = imbalance_ratio([7, 8, 16, 26], get_snet_stage_time)
optimed_tnet_ratio, optimed_tnet_stage_times = imbalance_ratio(tnet_split, get_tnet_stage_time)
default_tnet_ratio, default_tnet_stage_times = imbalance_ratio([8, 4, 4, 14], get_tnet_stage_time)

print(f"\n[Imbalance Ratio Test]")
print(f"Optimized SNet ratio \t | Per-stage Times(s)")
print(f"{optimed_snet_ratio:.3f} \t| {[f'{t:.3f}' for t in optimed_snet_stage_times]}")
print(f"{default_snet_ratio:.3f} \t| {[f'{t:.3f}' for t in default_snet_stage_times]}")
print(f"Optimized TNet ratio \t | Per-stage Times(s)")
print(f"{optimed_tnet_ratio:.3f} \t| {[f'{t:.3f}' for t in optimed_tnet_stage_times]}")
print(f"{default_tnet_ratio:.3f} \t| {[f'{t:.3f}' for t in default_tnet_stage_times]}")