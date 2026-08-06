"""
Compare SAC N=1 seed=0 learning curves: original [400,300] network
vs. shallow [128,128] network, to see whether the shallower network
changes the plateau/oscillation shape, even though final performance
is nearly identical.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_eval_curve(log_dir_glob):
    matches = glob.glob(log_dir_glob)
    if not matches:
        return [], []
    log_dir = matches[0]
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    if 'eval/mean_reward' not in ea.Tags()['scalars']:
        return [], []
    eval_rew = ea.Scalars('eval/mean_reward')
    return [x.step for x in eval_rew], [x.value for x in eval_rew]

orig_steps, orig_vals = load_eval_curve('m3_logs/sac_sweep/N1/SAC_seed0_1')
shallow_steps, shallow_vals = load_eval_curve('m3_logs/sac_n1_shallow_test/N1/SAC_seed0_1')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(orig_steps, orig_vals, marker='o', markersize=3, label='Original [400,300]', color='steelblue')
ax.plot(shallow_steps, shallow_vals, marker='s', markersize=3, label='Shallow [128,128]', color='darkorange')
ax.axvline(x=150000, color='red', linestyle='--', alpha=0.4, label='150k steps')
ax.axvline(x=200000, color='red', linestyle=':', alpha=0.4, label='200k steps')
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Eval Mean Reward ($)')
ax.set_title('SAC N=1 seed=0 — Network Architecture Comparison')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plotting/fig_architecture_comparison.png', dpi=150)
print("Saved: plotting/fig_architecture_comparison.png")

if orig_vals and shallow_vals:
    half_o = len(orig_vals) // 2
    half_s = len(shallow_vals) // 2
    print(f"\nOriginal [400,300]: final={orig_vals[-1]:.1f}, "
          f"2nd-half std={np.std(orig_vals[half_o:]):.2f}")
    print(f"Shallow [128,128]:  final={shallow_vals[-1]:.1f}, "
          f"2nd-half std={np.std(shallow_vals[half_s:]):.2f}")