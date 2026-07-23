"""
Compare the original SAC N=1 seed=0 learning curve (learning_rate=0.001)
against the LR-test run (learning_rate=0.0003), to check whether the
lower learning rate actually fixes the oscillation/degradation observed
after ~150-200k steps.
"""
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_eval_curve(log_dir_glob):
    log_dir = glob.glob(log_dir_glob)[0]
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    eval_rew = ea.Scalars('eval/mean_reward')
    steps = [x.step for x in eval_rew]
    vals = [x.value for x in eval_rew]
    return steps, vals

# Original run: learning_rate = 0.001
orig_steps, orig_vals = load_eval_curve('m3_logs/sac_sweep/N1/SAC_seed0_1')

# LR test run: learning_rate = 0.0003
test_steps, test_vals = load_eval_curve('m3_logs/sac_lr_test/N1/SAC_seed0_1')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(orig_steps, orig_vals, marker='o', markersize=3, label='Original (lr=0.001)', color='steelblue')
ax.plot(test_steps, test_vals, marker='s', markersize=3, label='LR test (lr=0.0003)', color='darkorange')
ax.axvline(x=200000, color='red', linestyle='--', alpha=0.4, label='200k steps')
ax.axvline(x=300000, color='red', linestyle=':', alpha=0.4, label='300k steps')
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Eval Mean Reward ($)')
ax.set_title('SAC N=1 seed=0 — Learning Rate Comparison')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plotting/fig_lr_comparison.png', dpi=150)
print("Saved: plotting/fig_lr_comparison.png")

print(f"\nOriginal (lr=0.001): first={orig_vals[0]:.1f} at step {orig_steps[0]}, "
      f"max={max(orig_vals):.1f} at step {orig_steps[orig_vals.index(max(orig_vals))]}, "
      f"last={orig_vals[-1]:.1f} at step {orig_steps[-1]}")
print(f"LR test (lr=0.0003): first={test_vals[0]:.1f} at step {test_steps[0]}, "
      f"max={max(test_vals):.1f} at step {test_steps[test_vals.index(max(test_vals))]}, "
      f"last={test_vals[-1]:.1f} at step {test_steps[-1]}")

# Compare variance in the second half of training (where oscillation was observed)
import numpy as np
half = len(orig_vals) // 2
orig_second_half_std = np.std(orig_vals[half:])
half_t = len(test_vals) // 2
test_second_half_std = np.std(test_vals[half_t:])
print(f"\nStd dev in second half of training:")
print(f"  Original (lr=0.001): {orig_second_half_std:.2f}")
print(f"  LR test (lr=0.0003): {test_second_half_std:.2f}")
