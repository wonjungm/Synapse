import re
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import os

BASE = os.path.dirname(__file__)
IN_CSV = os.path.join(BASE, 'checkpoint_summary.csv')
OUT_CSV = os.path.join(BASE, 'checkpoint_summary_sorted.csv')
OUT_PNG = os.path.join(BASE, 'checkpoint_summary_sorted.png')

if not os.path.exists(IN_CSV):
    raise SystemExit(f"Input CSV not found: {IN_CSV}")

df = pd.read_csv(IN_CSV)
# extract n value from experiment name (e.g., exp0_n100_...)

def extract_n(exp_name):
    m = re.search(r'n(\d+)', str(exp_name))
    return int(m.group(1)) if m else 10**9

df['n_val'] = df['experiment'].apply(extract_n)
df_sorted = df.sort_values('n_val')
# write sorted CSV
df_sorted.to_csv(OUT_CSV, index=False)

# prepare plot (same layout as original)
exps = df_sorted['experiment'].tolist()
avg_excl = df_sorted['avg_excluding_checkpoint_ms'].fillna(0).tolist()
avg_incl = df_sorted['avg_including_ms'].fillna(0).tolist()
mean_c = df_sorted['mean_C_save_s'].fillna(0).tolist()

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

axes[1].bar(ind, mean_c, color='C2')
axes[1].set_xticks(ind)
axes[1].set_xticklabels(exps, rotation=45, ha='right')
axes[1].set_ylabel('seconds')
axes[1].set_title('Mean checkpoint save duration (C_save)')

plt.tight_layout()
plt.savefig(OUT_PNG)
print('Wrote sorted CSV:', OUT_CSV)
print('Wrote sorted PNG:', OUT_PNG)
