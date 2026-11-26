"""
plot_n5_diagnostics.py

Diagnostic plot for the SAC N=5 preliminary cell: reward curve on top,
critic/actor loss and entropy coefficient on the bottom, all against the
same x-axis (training steps), so we can see exactly where (if anywhere)
the losses start blowing up relative to when the reward starts dropping.

Run from project root:
    python plot_n5_diagnostics.py
"""

import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = glob.glob('m3_logs/sac_sweep_prelim/N5/SAC_seed0_1')[0]
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

print("Available scalar tags:", ea.Tags()['scalars'])

def get(tag):
    try:
        s = ea.Scalars(tag)
        return [x.step for x in s], [x.value for x in s]
    except KeyError:
        return [], []

rollout_steps, rollout_vals = get('rollout/ep_rew_mean')
eval_steps, eval_vals       = get('eval/mean_reward')
critic_steps, critic_vals   = get('train/critic_loss')
actor_steps, actor_vals     = get('train/actor_loss')
ent_steps, ent_vals         = get('train/ent_coef')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

ax1.plot(rollout_steps, rollout_vals, alpha=0.4, label='rollout/ep_rew_mean (noisy)', color='steelblue')
if eval_vals:
    ax1.plot(eval_steps, eval_vals, marker='o', markersize=3, label='eval/mean_reward', color='darkorange', linewidth=2)
ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax1.set_ylabel('Return ($)')
ax1.set_title('SAC N=5 seed=0 -- Real CAISO -- Reward vs. Training Step')
ax1.legend()
ax1.grid(alpha=0.3)

if critic_vals:
    ax2.plot(critic_steps, critic_vals, label='train/critic_loss', color='crimson')
if actor_vals:
    ax2.plot(actor_steps, actor_vals, label='train/actor_loss', color='purple')
ax2.set_yscale('symlog')  # losses can go very large or negative -- symlog handles both
ax2.set_xlabel('Environment Steps')
ax2.set_ylabel('Loss (symlog scale)')
ax2.set_title('Critic / Actor Loss vs. Training Step')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('sac_n5_diagnostics.png', dpi=150)
print("Saved: sac_n5_diagnostics.png")

if rollout_vals:
    print(f"\nFirst rollout reward: {rollout_vals[0]:.1f} at step {rollout_steps[0]}")
    print(f"Last rollout reward:  {rollout_vals[-1]:.1f} at step {rollout_steps[-1]}")
    max_i = rollout_vals.index(max(rollout_vals))
    print(f"Max rollout reward:   {rollout_vals[max_i]:.1f} at step {rollout_steps[max_i]}")
    min_i = rollout_vals.index(min(rollout_vals))
    print(f"Min rollout reward:   {rollout_vals[min_i]:.1f} at step {rollout_steps[min_i]}")

if critic_vals:
    max_c = max(critic_vals)
    max_c_i = critic_vals.index(max_c)
    print(f"\nMax critic_loss: {max_c:.3e} at step {critic_steps[max_c_i]}")
    print(f"First critic_loss: {critic_vals[0]:.3e} at step {critic_steps[0]}")
    print(f"Last critic_loss:  {critic_vals[-1]:.3e} at step {critic_steps[-1]}")