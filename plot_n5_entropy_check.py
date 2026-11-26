"""
plot_n5_entropy_check.py

Follow-up to plot_n5_diagnostics.py. Checks whether SAC's auto-tuned
entropy coefficient (train/ent_coef) is oscillating in sync with the
eval/mean_reward swings seen at N=5 -- if so, that points to the entropy
target (which scales with N in SB3's default auto-tuning) as the likely
root cause of the non-convergence, rather than network capacity.

Run from project root:
    python plot_n5_entropy_check.py
"""

import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = glob.glob('m3_logs/sac_sweep_prelim/N5/SAC_seed0_1')[0]
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

def get(tag):
    s = ea.Scalars(tag)
    return [x.step for x in s], [x.value for x in s]

eval_steps, eval_vals = get('eval/mean_reward')
ent_steps, ent_vals = get('train/ent_coef')
ent_loss_steps, ent_loss_vals = get('train/ent_coef_loss')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(eval_steps, eval_vals, marker='o', markersize=4, color='darkorange', linewidth=2)
ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax1.set_ylabel('eval/mean_reward ($)')
ax1.set_title('SAC N=5 seed=0 -- Reward vs. Entropy Coefficient')
ax1.grid(alpha=0.3)

ax2.plot(ent_steps, ent_vals, color='teal', label='train/ent_coef')
ax2.set_ylabel('ent_coef', color='teal')
ax2.set_xlabel('Environment Steps')
ax2.grid(alpha=0.3)

ax2b = ax2.twinx()
ax2b.plot(ent_loss_steps, ent_loss_vals, color='gray', alpha=0.5, label='train/ent_coef_loss')
ax2b.set_ylabel('ent_coef_loss', color='gray')

plt.tight_layout()
plt.savefig('sac_n5_entropy_check.png', dpi=150)
print("Saved: sac_n5_entropy_check.png")

print(f"\nent_coef range: {min(ent_vals):.3f} to {max(ent_vals):.3f}")
print(f"ent_coef first/last: {ent_vals[0]:.3f} -> {ent_vals[-1]:.3f}")