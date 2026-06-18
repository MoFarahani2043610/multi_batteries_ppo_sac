"""
m2_scripts/generate_m2_plots.py
================================
Generates Milestone 2 deliverable figures:

  Figure M2a: Learning curves — SAC (5 seeds) with variance band
  Figure M2b: Learning curves — PPO (5 seeds) with variance band  
  Figure M2c: Combined SAC vs PPO learning curves
  
Each figure includes horizontal reference lines for:
  - Random policy baseline
  - Threshold policy baseline  
  - Perfect-foresight LP upper bound
  - 70% LP target (acceptance criterion)

Usage
-----
    python m2_scripts/generate_m2_plots.py

Outputs saved to: experiments/m2_learning_curves/
"""

from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _ROOT)

OUT_DIR  = os.path.join(_ROOT, 'experiments', 'm2_learning_curves')
LOG_BASE = os.path.join(_ROOT, 'm2_logs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Baselines (real CAISO data) ───────────────────────────────────
RANDOM_RETURN    = -12.72
THRESHOLD_RETURN =  86.82
LP_RETURN        = 182.83
TARGET_RETURN    = LP_RETURN * 0.70   # 127.98

# ── Colors ────────────────────────────────────────────────────────
SAC_COLOR  = '#1F4E79'
PPO_COLOR  = '#C55A11'
GREEN      = '#375623'
GRAY       = '#757575'
RED        = '#C00000'

N_SEEDS = 5


def load_eval_curves(algo: str) -> dict:
    """Load evaluation curves from evaluations.npz for all seeds."""
    curves = {}
    for seed in range(N_SEEDS):
        path = os.path.join(LOG_BASE, f'{algo}_n1', f'seed_{seed}',
                           'eval_results', 'evaluations.npz')
        if not os.path.exists(path):
            print(f'  Warning: {algo} seed={seed} eval not found')
            continue
        data = np.load(path)
        timesteps    = data['timesteps']
        mean_returns = data['results'].mean(axis=1)
        curves[seed] = {'timesteps': timesteps, 'returns': mean_returns}
    return curves


def plot_learning_curves(algo: str, curves: dict, save_path: str,
                          title: str, color: str) -> None:
    """Plot learning curves with mean ± std band across seeds."""
    if not curves:
        print(f'  No data for {algo} — skipping')
        return

    # align all seeds to common timestep grid
    all_timesteps = []
    for seed_data in curves.values():
        all_timesteps.extend(seed_data['timesteps'].tolist())
    timesteps = np.array(sorted(set(all_timesteps)))

    # interpolate each seed to common grid
    interp_returns = []
    for seed_data in curves.values():
        t  = seed_data['timesteps']
        r  = seed_data['returns']
        ri = np.interp(timesteps, t, r)
        interp_returns.append(ri)

    interp_returns = np.array(interp_returns)   # (n_seeds, T)
    mean_r = interp_returns.mean(axis=0)
    std_r  = interp_returns.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # individual seed lines (faint)
    for i, seed_data in enumerate(curves.values()):
        ax.plot(seed_data['timesteps'] / 1000, seed_data['returns'],
                color=color, alpha=0.2, linewidth=0.8)

    # mean line
    ax.plot(timesteps / 1000, mean_r, color=color, linewidth=2.5,
            label=f'{algo.upper()} mean (n={len(curves)} seeds)')

    # variance band
    ax.fill_between(timesteps / 1000,
                    mean_r - std_r,
                    mean_r + std_r,
                    color=color, alpha=0.15,
                    label=f'{algo.upper()} ± 1 std')

    # reference lines
    ax.axhline(LP_RETURN,        color=GREEN,  linewidth=1.5, linestyle='--',
               label=f'LP upper bound (${LP_RETURN:.2f})')
    ax.axhline(TARGET_RETURN,    color=GREEN,  linewidth=1.5, linestyle=':',
               label=f'Target 70% LP (${TARGET_RETURN:.2f})')
    ax.axhline(THRESHOLD_RETURN, color=GRAY,   linewidth=1.2, linestyle='--',
               label=f'Threshold (${THRESHOLD_RETURN:.2f})')
    ax.axhline(RANDOM_RETURN,    color=RED,    linewidth=1.0, linestyle='--',
               label=f'Random (${RANDOM_RETURN:.2f})')

    ax.set_xlabel('Environment Steps (K)', fontsize=12)
    ax.set_ylabel('Mean Episode Return ($)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, color='#CCCCCC', linestyle='--', alpha=0.7)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_combined(sac_curves: dict, ppo_curves: dict, save_path: str) -> None:
    """Plot SAC and PPO on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, curves, color in [
        ('SAC', sac_curves, SAC_COLOR),
        ('PPO', ppo_curves, PPO_COLOR),
    ]:
        if not curves:
            continue

        all_timesteps = []
        for sd in curves.values():
            all_timesteps.extend(sd['timesteps'].tolist())
        timesteps = np.array(sorted(set(all_timesteps)))

        interp_returns = []
        for sd in curves.values():
            ri = np.interp(timesteps, sd['timesteps'], sd['returns'])
            interp_returns.append(ri)

        interp_returns = np.array(interp_returns)
        mean_r = interp_returns.mean(axis=0)
        std_r  = interp_returns.std(axis=0)

        ax.plot(timesteps / 1000, mean_r, color=color, linewidth=2.5,
                label=f'{algo} mean (n={len(curves)} seeds)')
        ax.fill_between(timesteps / 1000,
                        mean_r - std_r, mean_r + std_r,
                        color=color, alpha=0.12)

    # reference lines
    ax.axhline(LP_RETURN,        color=GREEN, linewidth=1.5, linestyle='--',
               label=f'LP upper bound (${LP_RETURN:.2f})')
    ax.axhline(TARGET_RETURN,    color=GREEN, linewidth=1.5, linestyle=':',
               label=f'Target 70% LP (${TARGET_RETURN:.2f})')
    ax.axhline(THRESHOLD_RETURN, color=GRAY,  linewidth=1.2, linestyle='--',
               label=f'Threshold (${THRESHOLD_RETURN:.2f})')
    ax.axhline(RANDOM_RETURN,    color=RED,   linewidth=1.0, linestyle='--',
               label=f'Random (${RANDOM_RETURN:.2f})')

    ax.set_xlabel('Environment Steps (K)', fontsize=12)
    ax.set_ylabel('Mean Episode Return ($)', fontsize=12)
    ax.set_title('Figure M2 — SAC vs PPO Learning Curves at N=1 (Real CAISO Data)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, color='#CCCCCC', linestyle='--', alpha=0.7)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Milestone 2 — Learning Curve Generator')
    print('='*60)

    print('\nLoading SAC eval curves...')
    sac_curves = load_eval_curves('sac')
    print(f'  Found {len(sac_curves)} SAC seeds')

    print('Loading PPO eval curves...')
    ppo_curves = load_eval_curves('ppo')
    print(f'  Found {len(ppo_curves)} PPO seeds')

    print('\nGenerating figures...')

    plot_learning_curves(
        'sac', sac_curves,
        save_path=os.path.join(OUT_DIR, 'fig_m2a_sac_learning_curves.png'),
        title='Figure M2a — SAC Learning Curves at N=1 (5 Seeds, Real CAISO)',
        color=SAC_COLOR
    )

    plot_learning_curves(
        'ppo', ppo_curves,
        save_path=os.path.join(OUT_DIR, 'fig_m2b_ppo_learning_curves.png'),
        title='Figure M2b — PPO Learning Curves at N=1 (5 Seeds, Real CAISO)',
        color=PPO_COLOR
    )

    plot_combined(
        sac_curves, ppo_curves,
        save_path=os.path.join(OUT_DIR, 'fig_m2c_sac_vs_ppo_learning_curves.png')
    )

    print(f'\nAll figures saved to: {OUT_DIR}')
    print('='*60)