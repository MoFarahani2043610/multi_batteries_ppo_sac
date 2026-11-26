"""
plot_n5_long_diagnostics.py

Diagnostic plot for the 1,000,000-step SAC N=5 long training check:
reward curve on top, critic/actor loss below, so we can see whether
the reward curve was still climbing at the end or had flattened out.

Run from project root:
    python plot_n5_long_diagnostics.py
"""

import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = glob.glob('m3_logs/sac_n5_long_check/N5/SAC_seed0_1')[0]
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

ax1.plot(rollout_steps, rollout_vals, alpha=0.4, label='rollout/ep_rew_mean (noisy)', color='steelblue')
if eval_vals:
    ax1.plot(eval_steps, eval_vals, marker='o', markersize=3, label='eval/mean_reward', color='darkorange', linewidth=2)
ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax1.set_ylabel('Return ($)')
ax1.set_title('SAC N=5 seed=0 -- 1,000,000 steps -- Reward vs. Training Step')
ax1.legend()
ax1.grid(alpha=0.3)

if critic_vals:
    ax2.plot(critic_steps, critic_vals, label='train/critic_loss', color='crimson')
if actor_vals:
    ax2.plot(actor_steps, actor_vals, label='train/actor_loss', color='purple')
ax2.set_yscale('symlog')
ax2.set_xlabel('Environment Steps')
ax2.set_ylabel('Loss (symlog scale)')
ax2.set_title('Critic / Actor Loss vs. Training Step')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('sac_n5_long_diagnostics.png', dpi=150)
print("Saved: sac_n5_long_diagnostics.png")

if rollout_vals:
    # look at the trend over the final 20% of training to see if it's still climbing
    n = len(rollout_vals)
    tail_start = int(n * 0.8)
    print(f"\nFirst rollout reward: {rollout_vals[0]:.1f} at step {rollout_steps[0]}")
    print(f"Last rollout reward:  {rollout_vals[-1]:.1f} at step {rollout_steps[-1]}")
    print(f"Mean over final 20% of training: {sum(rollout_vals[tail_start:]) / len(rollout_vals[tail_start:]):.2f}")
    print(f"Mean over first 20% after warmup (20-40%): {sum(rollout_vals[int(n*0.2):int(n*0.4)]) / len(rollout_vals[int(n*0.2):int(n*0.4)]):.2f}")

if eval_vals:
    n = len(eval_vals)
    tail_start = max(1, int(n * 0.7))
    print(f"\nLast 30% of eval checkpoints: {[round(v,1) for v in eval_vals[tail_start:]]}")