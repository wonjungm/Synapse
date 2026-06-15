import json
import os
from typing import Dict, Optional, Tuple

import torch

def generate_alpha_beta():
    """
    Generate alpha_g and beta_g values for each GPU based on GPU properties.
    Returns:
        alpha_g (dict): Alpha values for each GPU.
        beta_g (dict): Beta values for each GPU.
    """
    num_gpus = torch.cuda.device_count()
    alpha_g = {}
    beta_g = {}

    for gpu_id in range(num_gpus):
        device = torch.device(f"cuda:{gpu_id}")
        properties = torch.cuda.get_device_properties(device)

        sm_count = float(getattr(properties, "multi_processor_count", 1.0))
        max_threads_per_sm = float(getattr(properties, "max_threads_per_multi_processor", 1024.0))
        total_memory = float(getattr(properties, "total_memory", 1.0))
        l2_cache_size = float(getattr(properties, "L2_cache_size", 1.0))

        # Use API-stable properties as lightweight proxies.
        compute_power = sm_count * max_threads_per_sm
        memory_bandwidth = total_memory + l2_cache_size

        # Normalize values to generate alpha and beta
        alpha_g[gpu_id] = compute_power / 1e5
        beta_g[gpu_id] = memory_bandwidth / 1e11

    return alpha_g, beta_g


def compute_and_save_alpha_beta(
    snet_csv: Optional[str] = None,
    tnet_csv: Optional[str] = None,
    out_path: Optional[str] = None,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute alpha/beta and persist them to JSON.

    Args:
        snet_csv: Reserved for compatibility with existing callers.
        tnet_csv: Reserved for compatibility with existing callers.
        out_path: Output JSON path. Defaults to ./alpha_beta_values.json.

    Returns:
        (alpha_g, beta_g) with int GPU ids as keys.
    """
    # Keep these arguments for API compatibility even though current generation
    # is hardware-property based rather than profile-CSV based.
    _ = snet_csv, tnet_csv

    alpha_g, beta_g = generate_alpha_beta()

    if out_path is None:
        out_path = os.path.abspath("alpha_beta_values.json")
    else:
        out_path = os.path.abspath(out_path)

    # JSON object keys are strings by spec; consumers should cast back to int.
    payload = {
        "alpha_g": {str(k): float(v) for k, v in alpha_g.items()},
        "beta_g": {str(k): float(v) for k, v in beta_g.items()},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return alpha_g, beta_g

if __name__ == "__main__":
    alpha_g, beta_g = generate_alpha_beta()
    print("Alpha values:", alpha_g)
    print("Beta values:", beta_g)