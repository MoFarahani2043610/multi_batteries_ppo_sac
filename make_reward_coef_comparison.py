"""
Comparison chart: all N=1 experiments (architecture and reward-coefficient
variants) for both SAC and PPO, seed 0.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── SAC panel ─────────────────────────────────────────────────────
sac_labels = ['Original\n[400,300]', 'Shallow\n[128,128]', 'Reward\nnormalized']
sac_values = [96.20, 97.18, 106.76]
sac_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars1 = axes[0].bar(sac_labels, sac_values, color=sac_colors, edgecolor='black', linewidth=0.6)
for bar, val in zip(bars1, sac_values):
    axes[0].annotate(f'${val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                      xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
axes[0].set_title('SAC $N=1$ (seed 0): Experiment Comparison')
axes[0].set_ylabel('Mean Return ($)')
axes[0].grid(axis='y', alpha=0.3)

# ── PPO panel ─────────────────────────────────────────────────────
ppo_labels = ['Original\n(M2, 5-seed mean)', 'Reward\nnormalized (seed 0)']
ppo_values = [98.12, 78.36]
ppo_colors = ['#1f77b4', '#d62728']

bars2 = axes[1].bar(ppo_labels, ppo_values, color=ppo_colors, edgecolor='black', linewidth=0.6)
for bar, val in zip(bars2, ppo_values):
    axes[1].annotate(f'${val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                      xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
axes[1].set_title('PPO $N=1$: Reward Normalization Effect')
axes[1].set_ylabel('Mean Return ($)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plotting/fig_reward_coef_comparison.png', dpi=150)
print("Saved: plotting/fig_reward_coef_comparison.png")