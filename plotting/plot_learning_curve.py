"""
Plot the real learning curve (ep_rew_mean and eval/mean_reward) from TensorBoard logs
to check whether training oscillates / stops improving after 200k-300k steps.
"""
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = glob.glob('m3_logs/sac_sweep/N1/SAC_seed0_1')[0]
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# rollout reward (noisy, per-episode average)
rollout = ea.Scalars('rollout/ep_rew_mean')
rollout_steps = [x.step for x in rollout]
rollout_vals = [x.value for x in rollout]

# eval reward (cleaner, evaluated periodically)
eval_rew = ea.Scalars('eval/mean_reward')
eval_steps = [x.step for x in eval_rew]
eval_vals = [x.value for x in eval_rew]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rollout_steps, rollout_vals, alpha=0.4, label='rollout/ep_rew_mean (noisy)', color='steelblue')
ax.plot(eval_steps, eval_vals, marker='o', markersize=3, label='eval/mean_reward', color='darkorange', linewidth=2)
ax.axvline(x=200000, color='red', linestyle='--', alpha=0.5, label='200k steps')
ax.axvline(x=300000, color='red', linestyle=':', alpha=0.5, label='300k steps')
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Return ($)')
ax.set_title('SAC N=1 seed=0 — Real CAISO — Learning Curve')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('sac_n1_seed0_learning_curve.png', dpi=150)
print("Saved: sac_n1_seed0_learning_curve.png")
print(f"Total eval points: {len(eval_vals)}")
print(f"First eval reward: {eval_vals[0]:.1f} at step {eval_steps[0]}")
print(f"Last eval reward: {eval_vals[-1]:.1f} at step {eval_steps[-1]}")
print(f"Max eval reward: {max(eval_vals):.1f} at step {eval_steps[eval_vals.index(max(eval_vals))]}")