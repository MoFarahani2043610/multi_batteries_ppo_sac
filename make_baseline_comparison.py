"""
Generate the baseline comparison figure: Random, Threshold, PPO, SAC
mean returns at N=1, compared against the perfect-foresight Oracle.
"""
import matplotlib.pyplot as plt
import numpy as np

labels = ['Random', 'Threshold', 'PPO', 'SAC', 'Oracle $V^*$']
values = [-12.72, 86.82, 98.12, 114.97, 182.83]
colors = ['#d62728', '#7f7f7f', '#ff7f0e', '#1f77b4', '#2ca02c']

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.6)

# annotate each bar with its value
for bar, val in zip(bars, values):
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    offset = 3 if height >= 0 else -3
    ax.annotate(f'${val:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, offset), textcoords='offset points',
                ha='center', va=va, fontsize=10)

# 70% target reference line
target = 127.98
ax.axhline(target, color='black', linestyle='--', linewidth=1, alpha=0.6)
ax.text(4.4, target + 3, '70% target ($127.98)', ha='right', fontsize=9, style='italic')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Mean Return ($/day)')
ax.set_title('$N=1$ Policy Comparison: Random, Threshold, PPO, SAC vs. Oracle')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plotting/fig_baseline_comparison.png', dpi=150)
print("Saved: plotting/fig_baseline_comparison.png")