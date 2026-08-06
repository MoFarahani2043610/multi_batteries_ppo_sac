"""
Full training diagnostics for N=1 SAC and PPO (all 5 seeds).
Generates the graphs Danial requested: training reward, evaluation reward,
actor/policy loss, critic/value loss, entropy (coefficient), and KL
divergence (PPO only).
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

# ── SAC: N=1, 5 seeds ──────────────────────────────────────────────
print("Loading SAC N=1 logs...")
sac_dirs = {
    0: 'm3_logs/sac_sweep/N1/SAC_seed0_1',
    1: 'm3_logs/sac_sweep/N1/SAC_seed1_1',
    2: 'm3_logs/sac_sweep/N1/SAC_seed2_1',
    3: 'm3_logs/sac_sweep/N1/SAC_seed3_1',
    4: 'm3_logs/sac_sweep/N1/SAC_seed4_1',
}
# fall back to glob if exact names differ
for seed in list(sac_dirs.keys()):
    if not glob.glob(sac_dirs[seed]):
        matches = find_seed_dirs(f'm3_logs/sac_sweep/N1/SAC_seed{seed}_*')
        if matches:
            sac_dirs[seed] = matches[0]
        else:
            print(f"  WARNING: no log dir found for SAC seed {seed}")

sac_curves = {}
for seed, d in sac_dirs.items():
    if not glob.glob(d):
        continue
    curves = {}
    for tag in ['rollout/ep_rew_mean', 'eval/mean_reward', 'train/actor_loss',
                'train/critic_loss', 'train/ent_coef']:
        steps, vals = load_scalar(d, tag)
        curves[tag] = (steps, vals)
    sac_curves[seed] = curves
    print(f"  seed {seed}: loaded {sum(len(v[0])>0 for v in curves.values())}/5 tags")

# ── Figure: SAC N=1 — all 5 seeds, 4-panel diagnostic ──────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0, 1, 5))

for seed, curves in sac_curves.items():
    c = colors[seed]
    steps, vals = curves['eval/mean_reward']
    if steps:
        axes[0,0].plot(steps, vals, color=c, alpha=0.7, label=f'seed {seed}')
    steps, vals = curves['train/actor_loss']
    if steps:
        axes[0,1].plot(steps, vals, color=c, alpha=0.5)
    steps, vals = curves['train/critic_loss']
    if steps:
        axes[1,0].plot(steps, vals, color=c, alpha=0.5)
    steps, vals = curves['train/ent_coef']
    if steps:
        axes[1,1].plot(steps, vals, color=c, alpha=0.7)

axes[0,0].set_title('Evaluation Reward'); axes[0,0].set_xlabel('Steps'); axes[0,0].set_ylabel('$')
axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)
axes[0,1].set_title('Actor Loss'); axes[0,1].set_xlabel('Steps'); axes[0,1].grid(alpha=0.3)
axes[1,0].set_title('Critic Loss'); axes[1,0].set_xlabel('Steps'); axes[1,0].grid(alpha=0.3)
axes[1,1].set_title('Entropy Coefficient (alpha)'); axes[1,1].set_xlabel('Steps'); axes[1,1].grid(alpha=0.3)
axes[1,1].text(0.02, 0.97, 'Target entropy = -1 (SB3 auto default, N=1).\nAlpha (this plot) is tuned so policy entropy\ntracks that target; SB3 does not log a\nseparate "target" scalar for SAC.',
               transform=axes[1,1].transAxes, fontsize=7, va='top', style='italic')

fig.suptitle('SAC N=1 Training Diagnostics (5 seeds)', fontsize=14)
plt.tight_layout()
plt.savefig('plotting/fig_sac_n1_diagnostics.png', dpi=150)
plt.close()
print("Saved: plotting/fig_sac_n1_diagnostics.png")

# ── PPO: N=1, 5 seeds ──────────────────────────────────────────────
print("\nLoading PPO N=1 logs...")
ppo_dirs = {}
for seed in range(5):
    matches = find_seed_dirs(f'm2_logs/ppo_n1/seed_{seed}/PPO_seed{seed}_*') or \
              find_seed_dirs(f'm2_logs/ppo_n1/PPO_seed{seed}_*')
    if matches:
        ppo_dirs[seed] = matches[0]
    else:
        print(f"  WARNING: no log dir found for PPO seed {seed}, trying alternate path")

ppo_curves = {}
for seed, d in ppo_dirs.items():
    curves = {}
    for tag in ['rollout/ep_rew_mean', 'eval/mean_reward', 'train/policy_gradient_loss',
                'train/value_loss', 'train/entropy_loss', 'train/approx_kl']:
        steps, vals = load_scalar(d, tag)
        curves[tag] = (steps, vals)
    ppo_curves[seed] = curves
    print(f"  seed {seed}: loaded {sum(len(v[0])>0 for v in curves.values())}/6 tags")

if ppo_curves:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for seed, curves in ppo_curves.items():
        c = colors[seed]
        for ax, tag in zip(axes.flat, ['eval/mean_reward', 'train/policy_gradient_loss',
                                         'train/value_loss', 'train/entropy_loss',
                                         'train/approx_kl']):
            steps, vals = curves[tag]
            if steps:
                ax.plot(steps, vals, color=c, alpha=0.6, label=f'seed {seed}' if tag=='eval/mean_reward' else None)

    titles = ['Evaluation Reward', 'Policy Gradient Loss', 'Value Loss', 'Entropy Loss', 'Approx KL Divergence']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.set_xlabel('Steps'); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].axis('off')

    fig.suptitle('PPO N=1 Training Diagnostics (5 seeds)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plotting/fig_ppo_n1_diagnostics.png', dpi=150)
    plt.close()
    print("Saved: plotting/fig_ppo_n1_diagnostics.png")
else:
    print("No PPO N=1 logs found — check m2_logs/ppo_n1/ path structure.")

print("\nDone.")