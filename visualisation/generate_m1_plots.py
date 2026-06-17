"""
visualisation/generate_m1_plots.py
====================================
Generates all Milestone 1 baseline visualisation plots.

Produces:
  - experiments/m1_baselines/random_episode.png
  - experiments/m1_baselines/threshold_episode.png
  - experiments/m1_baselines/lp_episode.png
  - experiments/m1_baselines/comparison_profit.png
  - experiments/m1_baselines/baseline_soc.png

Usage
-----
    python visualisation/generate_m1_plots.py
"""

from __future__ import annotations
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_ENV  = os.path.join(_ROOT, 'env')
_BASE = os.path.join(_ROOT, 'baselines')

for p in [_ROOT, _ENV, _BASE]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from storage_arbitrage_env import StorageArbitrageEnv
from random_policy    import RandomPolicy,    run_episodes as run_random
from threshold_policy import ThresholdPolicy, run_episodes as run_threshold

OUT_DIR = os.path.join(_ROOT, 'experiments', 'm1_baselines')
os.makedirs(OUT_DIR, exist_ok=True)

SAC_COLOR  = '#1F4E79'
PPO_COLOR  = '#C55A11'
GREEN      = '#375623'
GRAY       = '#757575'
ORANGE     = '#FF9800'
RED        = '#F44336'
BLUE       = '#2196F3'
PURPLE     = '#9C27B0'


def make_env(seed=0):
    return StorageArbitrageEnv(n_batteries=1)


def run_lp_episode(env, seed=0):
    """Run perfect foresight LP for one episode."""
    try:
        from perfect_foresight import PerfectForesightLP, _get_episode_prices
        solver = PerfectForesightLP(env)
        obs, info = env.reset(seed=seed)
        prices = _get_episode_prices(env)
        result = solver.solve(prices, info['soc_mwh'])

        # reconstruct episode arrays for plotting
        all_soc     = [info['soc_mwh'].copy()]
        all_prices  = [info['prices'].copy()]
        all_actions = []
        all_rewards = []

        soc = info['soc_mwh'].copy()
        lp_actions = result['action']  # shape (T, N)
        for t in range(len(lp_actions)):
            action_arr = np.atleast_1d(lp_actions[t]).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action_arr)
            all_soc.append(info['soc_mwh'].copy())
            all_prices.append(info['prices'].copy())
            all_actions.append(action_arr)
            all_rewards.append(reward)
            if terminated or truncated:
                break

        return {
            'all_soc':     all_soc,
            'all_prices':  all_prices,
            'all_actions': all_actions,
            'all_rewards': all_rewards,
            'profits':     [sum(all_rewards)],
        }
    except Exception as e:
        print(f'  LP episode failed: {e}')
        return None


def extract_arrays(res):
    """Extract numpy arrays from episode result."""
    soc     = np.stack([np.atleast_1d(s) for s in res['all_soc']])
    prices  = np.stack([np.atleast_1d(p) for p in res['all_prices']])
    actions = np.stack([np.atleast_1d(a) for a in res['all_actions']])
    rewards = np.array(res['all_rewards'])
    cum_profit = np.cumsum(rewards)
    T = len(rewards)
    t_hrs = np.arange(T + 1) * (5/60)
    t_hrs_r = t_hrs[:-1]
    return soc, prices, actions, rewards, cum_profit, t_hrs, t_hrs_r


def plot_episode_dashboard(res, title, save_path):
    """4-panel episode dashboard."""
    soc, prices, actions, rewards, cum_profit, t_hrs, t_hrs_r = extract_arrays(res)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(4, 1, hspace=0.45, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # Panel 1: Price
    p = prices[:, 0]
    ax1.plot(t_hrs, p, color=ORANGE, linewidth=1.5)
    ax1.fill_between(t_hrs, p.min(), p, color=ORANGE, alpha=0.12)
    mean_p = p.mean()
    spike_idx = np.where(p > 2 * mean_p)[0]
    if len(spike_idx):
        ax1.scatter(t_hrs[spike_idx], p[spike_idx], color='red', s=20, zorder=5, label='Spike')
        ax1.legend(fontsize=8)
    ax1.set_ylabel('Price ($/MWh)')
    ax1.set_title('Electricity Price (CAISO)')
    ax1.set_xlim(t_hrs[0], t_hrs[-1])
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    # Panel 2: SoC
    ax2.plot(t_hrs, soc[:, 0], color=SAC_COLOR, linewidth=1.8)
    ax2.fill_between(t_hrs, 0, soc[:, 0], color=SAC_COLOR, alpha=0.15)
    ax2.set_ylabel('SoC (MWh)')
    ax2.set_title('State of Charge')
    ax2.set_xlim(t_hrs[0], t_hrs[-1])
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    # Panel 3: Actions
    a = actions[:, 0]
    charge = np.where(a > 0, a, 0)
    disch  = np.where(a < 0, a, 0)
    ax3.bar(t_hrs_r, charge, width=5/60*0.8, color=GREEN,  alpha=0.8, label='Charge')
    ax3.bar(t_hrs_r, disch,  width=5/60*0.8, color=RED,    alpha=0.8, label='Discharge')
    ax3.axhline(0, color='gray', linewidth=0.6)
    ax3.set_ylabel('Power (MW)')
    ax3.set_title('Actions  (+ charge, − discharge)')
    ax3.set_xlim(t_hrs[0], t_hrs[-1])
    ax3.grid(True, alpha=0.2)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.set_xticklabels([])

    # Panel 4: Cumulative profit
    ax4.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax4.plot(t_hrs_r, cum_profit, color=BLUE, linewidth=2.0)
    ax4.fill_between(t_hrs_r, 0, cum_profit,
                     where=cum_profit >= 0, color=GREEN, alpha=0.15)
    ax4.fill_between(t_hrs_r, 0, cum_profit,
                     where=cum_profit < 0,  color=RED,   alpha=0.15)
    final = cum_profit[-1] if len(cum_profit) else 0
    ax4.set_ylabel('Profit ($)')
    ax4.set_xlabel('Time (hours)')
    ax4.set_title(f'Cumulative Profit  (final: ${final:.2f})')
    ax4.set_xlim(t_hrs_r[0], t_hrs_r[-1])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_comparison(results_dict, save_path):
    """Compare cumulative profit curves from all baseline policies."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title('Figure M1 — Cumulative Profit Comparison: All Baseline Policies',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    palette = {
        'Random':    GRAY,
        'Threshold': ORANGE,
        'LP (Upper Bound)': GREEN,
    }
    styles = {
        'Random':    '-',
        'Threshold': '-',
        'LP (Upper Bound)': '--',
    }

    for name, res in results_dict.items():
        _, _, _, rewards, cum_profit, _, t_hrs_r = extract_arrays(res)
        color = palette.get(name, BLUE)
        style = styles.get(name, '-')
        ax.plot(t_hrs_r, cum_profit, linewidth=2.0, linestyle=style,
                color=color, label=f'{name}  (${cum_profit[-1]:.2f})')

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, color='#CCCCCC', linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_soc_comparison(results_dict, save_path):
    """Compare SoC trajectories across policies."""
    fig, axes = plt.subplots(1, len(results_dict), figsize=(14, 4), sharey=True)
    fig.suptitle('Figure M1 — State of Charge Trajectories by Policy',
                 fontsize=13, fontweight='bold')

    palette = {'Random': GRAY, 'Threshold': ORANGE, 'LP (Upper Bound)': GREEN}

    for ax, (name, res) in zip(axes, results_dict.items()):
        soc, prices, _, _, _, t_hrs, _ = extract_arrays(res)
        color = palette.get(name, BLUE)
        ax.plot(t_hrs, soc[:, 0], color=color, linewidth=2.0)
        ax.fill_between(t_hrs, 0, soc[:, 0], color=color, alpha=0.15)
        profit = sum(res['all_rewards'])
        ax.set_title(f'{name}\n(${profit:.2f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('SoC (MWh)' if ax == axes[0] else '')
        ax.set_xlim(t_hrs[0], t_hrs[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Milestone 1 — Baseline Visualisations')
    print('='*60)

    SEED = 0
    env  = make_env(SEED)

    # ── Run episodes ──────────────────────────────────────────
    print('\nRunning Random Policy...')
    rp      = RandomPolicy(env, seed=SEED)
    res_r   = run_random(rp, env, n_episodes=1, seed=SEED)

    print('Running Threshold Policy...')
    tp      = ThresholdPolicy()
    tp.reset(env=env)   # required before act()
    res_t   = run_threshold(tp, env, n_episodes=1, seed=SEED)

    print('Running LP Policy...')
    res_lp  = run_lp_episode(env, seed=SEED)

    # ── Individual dashboards ─────────────────────────────────
    print('\nGenerating episode dashboards...')
    plot_episode_dashboard(res_r,
        title='Figure M1a — Random Policy Episode Dashboard (N=1, CAISO)',
        save_path=os.path.join(OUT_DIR, 'fig_m1a_random_episode.png'))

    plot_episode_dashboard(res_t,
        title='Figure M1b — Threshold Policy Episode Dashboard (N=1, CAISO)',
        save_path=os.path.join(OUT_DIR, 'fig_m1b_threshold_episode.png'))

    if res_lp:
        plot_episode_dashboard(res_lp,
            title='Figure M1c — Perfect-Foresight LP Episode Dashboard (N=1, CAISO)',
            save_path=os.path.join(OUT_DIR, 'fig_m1c_lp_episode.png'))

    # ── Comparison plots ──────────────────────────────────────
    print('\nGenerating comparison plots...')
    results_all = {'Random': res_r, 'Threshold': res_t}
    if res_lp:
        results_all['LP (Upper Bound)'] = res_lp

    plot_comparison(results_all,
        save_path=os.path.join(OUT_DIR, 'fig_m1d_comparison_profit.png'))

    plot_soc_comparison(results_all,
        save_path=os.path.join(OUT_DIR, 'fig_m1e_soc_comparison.png'))

    print(f'\nAll M1 plots saved to: {OUT_DIR}')
    print('='*60)