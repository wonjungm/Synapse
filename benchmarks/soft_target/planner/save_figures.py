# save_figures.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) Paths
# -----------------------------
PROFILE_DIR = "./profile"
OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

snet_path = os.path.join(PROFILE_DIR, "snet.csv")
tnet_path = os.path.join(PROFILE_DIR, "tnet.csv")

snet = pd.read_csv(snet_path)
tnet = pd.read_csv(tnet_path)

# layer index 숫자로 뽑기: "layer0" -> 0
def layer_to_int(x: str) -> int:
    return int(str(x).replace("layer", ""))

snet["layer_idx"] = snet["layer"].apply(layer_to_int)
tnet["layer_idx"] = tnet["layer"].apply(layer_to_int)

# -----------------------------
# 1) Layer-wise time plots
# -----------------------------
# (A) SNet: forward + backward
plt.figure(figsize=(10, 4))
plt.plot(snet["layer_idx"], snet["forward_time_ms"], label="SNet forward (ms)")
plt.plot(snet["layer_idx"], snet["backward_time_ms"], label="SNet backward (ms)")
plt.xlabel("Layer index")
plt.ylabel("Time (ms)")
plt.title("SNet layer-wise compute time")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "snet_layer_time.png"), dpi=200)
plt.close()

# (B) TNet: forward only (backward가 0이라 forward 중심)
plt.figure(figsize=(10, 4))
plt.plot(tnet["layer_idx"], tnet["forward_time_ms"], label="TNet forward (ms)")
plt.xlabel("Layer index")
plt.ylabel("Time (ms)")
plt.title("TNet layer-wise compute time (forward)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tnet_layer_time.png"), dpi=200)
plt.close()

# -----------------------------
# 2) Stage time comparison (Optimized vs Default)
#    (너 optimizer 출력 그대로)
# -----------------------------
snet_times_optim = [0.266, 0.278, 0.269, 0.281]
snet_times_default = [0.220, 0.276, 0.298, 0.373]

tnet_times_optim = [0.410, 0.426, 0.428, 0.429]
tnet_times_default = [0.341, 0.288, 0.288, 0.775]

stages = [0, 1, 2, 3]
bar_w = 0.35

# (A) SNet stage times
plt.figure(figsize=(7, 4))
x1 = [s - bar_w/2 for s in stages]
x2 = [s + bar_w/2 for s in stages]
plt.bar(x1, snet_times_default, width=bar_w, label="Default")
plt.bar(x2, snet_times_optim, width=bar_w, label="Optimized")
plt.xticks(stages)
plt.xlabel("Stage")
plt.ylabel("Time (s)")
plt.title("SNet per-stage time: Default vs Optimized")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "snet_stage_time_compare.png"), dpi=200)
plt.close()

# (B) TNet stage times
plt.figure(figsize=(7, 4))
plt.bar(x1, tnet_times_default, width=bar_w, label="Default")
plt.bar(x2, tnet_times_optim, width=bar_w, label="Optimized")
plt.xticks(stages)
plt.xlabel("Stage")
plt.ylabel("Time (s)")
plt.title("TNet per-stage time: Default vs Optimized")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tnet_stage_time_compare.png"), dpi=200)
plt.close()

# -----------------------------
# 3) Imbalance ratio summary bar
# -----------------------------
imbalance = pd.DataFrame({
    "model": ["SNet", "SNet", "TNet", "TNet"],
    "setting": ["Default", "Optimized", "Default", "Optimized"],
    "ratio": [1.279, 1.027, 1.832, 1.014]
})

plt.figure(figsize=(7, 4))
labels = [f"{m}\n{st}" for m, st in zip(imbalance["model"], imbalance["setting"])]
plt.bar(labels, imbalance["ratio"])
plt.axhline(1.0, linestyle="--")  # 1이 이상적 기준선
plt.ylabel("Imbalance ratio (worst/mean)")
plt.title("Stage-time imbalance ratio (lower is better, ideal≈1.0)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "imbalance_ratio.png"), dpi=200)
plt.close()

# -----------------------------
# 4) Stage memory plot (SNet vs TNet) + device limit line
# -----------------------------
# KB 단위 그대로 사용 (너 로그 그대로)
device_total_kb = 16228416.00

snet_mem_kb = [3345648.25, 2841404.0, 2440640.0, 2487740.796875]
tnet_mem_kb = [5608841.75, 5473713.75, 5473713.75, 5507402.546875]

plt.figure(figsize=(7, 4))
plt.bar([s - bar_w/2 for s in stages], snet_mem_kb, width=bar_w, label="SNet memory (KB)")
plt.bar([s + bar_w/2 for s in stages], tnet_mem_kb, width=bar_w, label="TNet memory (KB)")
plt.axhline(device_total_kb, linestyle="--", label="Device total memory (KB)")
plt.xticks(stages)
plt.xlabel("Stage")
plt.ylabel("Estimated memory (KB)")
plt.title("Per-stage memory feasibility check")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "stage_memory.png"), dpi=200)
plt.close()

# -----------------------------
# 5) Save a small summary table as CSV (보고서 표로 바로 쓰기)
# -----------------------------
summary = pd.DataFrame({
    "item": [
        "SNet partition", "TNet partition",
        "SNet imbalance default", "SNet imbalance optimized",
        "TNet imbalance default", "TNet imbalance optimized",
        "DP duration (ms)"
    ],
    "value": [
        "[9, 12, 16, 20]", "[9, 6, 6, 9]",
        "1.279", "1.027",
        "1.832", "1.014",
        "50.286"
    ]
})
summary.to_csv(os.path.join(OUT_DIR, "planner_summary_table.csv"), index=False)

print("[OK] Saved figures to:", OUT_DIR)
print(" - snet_layer_time.png")
print(" - tnet_layer_time.png")
print(" - snet_stage_time_compare.png")
print(" - tnet_stage_time_compare.png")
print(" - imbalance_ratio.png")
print(" - stage_memory.png")
print(" - planner_summary_table.csv")
