"""
Supplementary figure: SAC N=1 training diagnostics, one row per seed,
for unambiguous inspection of each seed individually (addresses
possible ambiguity in the combined overlapping-lines figure).
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_scalar(log_dir, tag):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        return [], []
    data = ea.Scalars(tag)
    return [x.step for x in data], [x.value for x in data]

def find_seed_dirs(pattern):
    return sorted(glob.glob(pattern))

sac_dirs = {}
for seed in range(5):
    matches = find_seed_dirs(f'm3_logs/sac_sweep/N1/SAC_seed{seed}_*')
    if matches:
        sac_dirs[seed] = matches[0]
    else:
        print(f"WARNING: no log dir found for SAC seed {seed}")

fig, axes = plt.subplots(5, 4, figsize=(16, 18))
metrics = ['eval/mean_reward', 'train/actor_loss', 'train/critic_loss', 'train/ent_coef']
titles = ['Evaluation Reward', 'Actor Loss', 'Critic Loss', 'Entropy Coefficient (alpha)']

for row, seed in enumerate(sorted(sac_dirs.keys())):
    d = sac_dirs[seed]
    for col, (tag, title) in enumerate(zip(metrics, titles)):
        steps, vals = load_scalar(d, tag)
        ax = axes[row, col]
        if steps:
            ax.plot(steps, vals, color=f'C{seed}', linewidth=1)
        if row == 0:
            ax.set_title(title, fontsize=11)
        if col == 0:
            ax.set_ylabel(f'seed {seed}', fontsize=11, fontweight='bold')
        if row == 4:
            ax.set_xlabel('Steps')
        ax.grid(alpha=0.3)

fig.suptitle('SAC N=1 Training Diagnostics — Per-Seed Detail', fontsize=15)
plt.tight_layout()
plt.savefig('plotting/fig_sac_n1_per_seed.png', dpi=150)
print("Saved: plotting/fig_sac_n1_per_seed.png")