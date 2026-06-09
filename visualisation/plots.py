"""
visualisation/plots.py
======================
Task 4 — Basic visualisations for Milestone 1.

Four plots:
  1. plot_soc()          — State of Charge over time, all batteries
  2. plot_profit()       — Cumulative profit over time
  3. plot_actions()      — Action histogram (charge / discharge distribution)
  4. plot_episode()      — Full episode dashboard: price + SoC + actions + profit

Usage
-----
    from visualisation.plots import plot_episode
    from baselines.random_policy import RandomPolicy, run_episodes
    from storage_arbitrage_env import StorageArbitrageEnv

    env    = StorageArbitrageEnv(n_batteries=1)
    policy = RandomPolicy(env, seed=0)
    res    = run_episodes(policy, env, n_episodes=1, seed=0)
    plot_episode(res, title="Random Policy — 1 Battery")

Install
-------
    pip install matplotlib
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "env")))

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")


def _extract_arrays(res):
    """
    Pull numpy arrays from a run_episodes() result dict.
    Handles both our env version and the student's env version.
    """
    # SoC — shape (T+1, N)
    soc_list = res["all_soc"]
    if isinstance(soc_list[0], np.ndarray):
        soc = np.stack(soc_list)                    # (T+1, N)
    else:
        soc = np.array([[s] for s in soc_list])     # (T+1, 1)

    # prices — shape (T+1, M)
    price_list = res["all_prices"]
    if isinstance(price_list[0], np.ndarray):
        prices = np.stack(price_list)
    else:
        prices = np.array([[p] for p in price_list])

    # actions — shape (T, N)
    act_list = res["all_actions"]
    if isinstance(act_list[0], np.ndarray):
        actions = np.stack(act_list)
    else:
        actions = np.array([[a] for a in act_list])

    # rewards — shape (T,)
    rewards = np.array(res["all_rewards"])

    # cumulative profit
    cum_profit = np.cumsum(rewards)

    # time axis in hours (5-min steps)
    T      = len(rewards)
    t_hrs  = np.arange(T + 1) * (5 / 60)
    t_hrs_r = t_hrs[:-1]   # for rewards / actions (length T)

    return soc, prices, actions, rewards, cum_profit, t_hrs, t_hrs_r


# ─────────────────────────────────────────────────────────────
#  1. SoC over time
# ─────────────────────────────────────────────────────────────

def plot_soc(res, title="State of Charge over Time", ax=None, save_path=None):
    """
    Plot SoC trajectory for all batteries over one episode.

    Parameters
    ----------
    res       : dict from run_episodes()
    title     : plot title
    ax        : existing matplotlib Axes (optional)
    save_path : if given, save figure to this path
    """
    _check_mpl()
    soc, prices, actions, rewards, cum_profit, t_hrs, _ = _extract_arrays(res)

    N       = soc.shape[1]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 4))

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(N, 1)))
    for i in range(N):
        ax.plot(t_hrs, soc[:, i], color=colors[i], linewidth=1.8,
                label=f"Battery {i+1}")
        ax.fill_between(t_hrs, 0, soc[:, i], color=colors[i], alpha=0.08)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("State of Charge (MWh)")
    ax.set_title(title)
    ax.set_xlim(t_hrs[0], t_hrs[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    if N > 1:
        ax.legend(loc="upper right", fontsize=8)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()


# ─────────────────────────────────────────────────────────────
#  2. Cumulative profit over time
# ─────────────────────────────────────────────────────────────

def plot_profit(res, title="Cumulative Profit over Time", ax=None, save_path=None):
    """
    Plot cumulative profit trajectory over one episode.
    """
    _check_mpl()
    soc, prices, actions, rewards, cum_profit, t_hrs, t_hrs_r = _extract_arrays(res)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 4))

    # colour line green where profit is positive, red where negative
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.plot(t_hrs_r, cum_profit, color="#2196F3", linewidth=2.0, label="Cumulative profit")
    ax.fill_between(t_hrs_r, 0, cum_profit,
                    where=cum_profit >= 0, color="#4CAF50", alpha=0.15, label="Profit")
    ax.fill_between(t_hrs_r, 0, cum_profit,
                    where=cum_profit < 0,  color="#F44336", alpha=0.15, label="Loss")

    final = cum_profit[-1] if len(cum_profit) else 0
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.set_title(f"{title}  (final: ${final:.2f})")
    ax.set_xlim(t_hrs_r[0], t_hrs_r[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()


# ─────────────────────────────────────────────────────────────
#  3. Action histogram
# ─────────────────────────────────────────────────────────────

def plot_actions(res, title="Action Distribution", ax=None, save_path=None):
    """
    Histogram of actions split into charge (positive) and discharge (negative).
    One panel per battery if N > 1.
    """
    _check_mpl()
    soc, prices, actions, rewards, cum_profit, t_hrs, _ = _extract_arrays(res)

    N       = actions.shape[1]
    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(1, N, figsize=(5 * N, 4), squeeze=False)
        axes = axes[0]
    else:
        axes = [ax] * N

    for i in range(N):
        a    = actions[:, i]
        ax_i = axes[i]

        charge    = a[a > 0]
        discharge = -a[a < 0]   # flip sign so both are positive for histogram
        idle_frac = np.mean(np.abs(a) < 1e-6) * 100

        bins = np.linspace(0, max(a.max(), 0.01), 25)
        ax_i.hist(charge,    bins=bins, color="#4CAF50", alpha=0.7, label="Charge")
        ax_i.hist(discharge, bins=bins, color="#F44336", alpha=0.7, label="Discharge")

        ax_i.set_xlabel("Power (MW)")
        ax_i.set_ylabel("Count")
        label = f"Battery {i+1}" if N > 1 else ""
        ax_i.set_title(f"{title} {label}\n(idle {idle_frac:.1f}% of steps)")
        ax_i.legend(fontsize=8)
        ax_i.grid(True, alpha=0.3)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()


# ─────────────────────────────────────────────────────────────
#  4. Full episode dashboard
# ─────────────────────────────────────────────────────────────

def plot_episode(res, title="Episode Dashboard", save_path=None):
    """
    4-panel dashboard for one episode:
      Row 1: electricity price over time
      Row 2: SoC over time (all batteries)
      Row 3: actions over time (charge/discharge)
      Row 4: cumulative profit over time

    Parameters
    ----------
    res       : dict from run_episodes()
    title     : overall figure title
    save_path : if given, save figure to this path
    """
    _check_mpl()
    soc, prices, actions, rewards, cum_profit, t_hrs, t_hrs_r = _extract_arrays(res)

    N = soc.shape[1]

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(4, 1, hspace=0.45, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # ── Panel 1: Price ───────────────────────────────────────
    p = prices[:, 0]               # first market node
    ax1.plot(t_hrs, p, color="#FF9800", linewidth=1.5)
    ax1.fill_between(t_hrs, p.min(), p, color="#FF9800", alpha=0.12)
    ax1.set_ylabel("Price ($/MWh)")
    ax1.set_title("Electricity Price")
    ax1.set_xlim(t_hrs[0], t_hrs[-1])
    ax1.grid(True, alpha=0.3)

    # mark price spikes (>2x mean)
    mean_p = p.mean()
    spike_idx = np.where(p > 2 * mean_p)[0]
    if len(spike_idx):
        ax1.scatter(t_hrs[spike_idx], p[spike_idx],
                    color="red", s=20, zorder=5, label="Spike")
        ax1.legend(fontsize=8)

    # ── Panel 2: SoC ─────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(N, 1)))
    for i in range(N):
        ax2.plot(t_hrs, soc[:, i], color=colors[i],
                 linewidth=1.8, label=f"Battery {i+1}")
        ax2.fill_between(t_hrs, 0, soc[:, i], color=colors[i], alpha=0.08)
    ax2.set_ylabel("SoC (MWh)")
    ax2.set_title("State of Charge")
    ax2.set_xlim(t_hrs[0], t_hrs[-1])
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    if N > 1:
        ax2.legend(loc="upper right", fontsize=7)

    # ── Panel 3: Actions ─────────────────────────────────────
    for i in range(N):
        a      = actions[:, i]
        charge = np.where(a > 0, a, 0)
        disch  = np.where(a < 0, a, 0)
        offset = i * 0.02          # slight offset for multi-battery readability
        ax3.bar(t_hrs_r + offset, charge, width=5/60 * 0.8,
                color="#4CAF50", alpha=0.7,
                label="Charge" if i == 0 else "")
        ax3.bar(t_hrs_r + offset, disch, width=5/60 * 0.8,
                color="#F44336", alpha=0.7,
                label="Discharge" if i == 0 else "")

    ax3.axhline(0, color="gray", linewidth=0.6)
    ax3.set_ylabel("Power (MW)")
    ax3.set_title("Actions  (+ charge, - discharge)")
    ax3.set_xlim(t_hrs[0], t_hrs[-1])
    ax3.grid(True, alpha=0.2)
    ax3.legend(fontsize=8, loc="upper right")

    # ── Panel 4: Cumulative Profit ───────────────────────────
    ax4.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax4.plot(t_hrs_r, cum_profit, color="#2196F3", linewidth=2.0)
    ax4.fill_between(t_hrs_r, 0, cum_profit,
                     where=cum_profit >= 0, color="#4CAF50", alpha=0.15)
    ax4.fill_between(t_hrs_r, 0, cum_profit,
                     where=cum_profit < 0,  color="#F44336", alpha=0.15)
    final = cum_profit[-1] if len(cum_profit) else 0
    ax4.set_ylabel("Profit ($)")
    ax4.set_xlabel("Time (hours)")
    ax4.set_title(f"Cumulative Profit  (final: ${final:.2f})")
    ax4.set_xlim(t_hrs_r[0], t_hrs_r[-1])
    ax4.grid(True, alpha=0.3)

    # shared x-axis label
    for ax in [ax1, ax2, ax3]:
        ax.set_xticklabels([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  5. Compare policies side by side
# ─────────────────────────────────────────────────────────────

def plot_comparison(results_dict, save_path=None):
    """
    Compare cumulative profit curves from multiple policies on one plot.

    Parameters
    ----------
    results_dict : dict mapping policy name -> run_episodes() result
                   e.g. {"Random": res_r, "Threshold": res_t, "LP": res_lp}
    save_path    : optional save path
    """
    _check_mpl()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title("Cumulative Profit Comparison", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    palette = {
        "Random":    "#9E9E9E",
        "Threshold": "#FF9800",
        "LP":        "#4CAF50",
        "SAC":       "#2196F3",
        "PPO":       "#9C27B0",
    }

    for name, res in results_dict.items():
        _, _, _, rewards, cum_profit, _, t_hrs_r = _extract_arrays(res)
        color = palette.get(name, "#607D8B")
        ax.plot(t_hrs_r, cum_profit, linewidth=2.0,
                color=color, label=f"{name}  (${cum_profit[-1]:.2f})")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  CLI — quick demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from storage_arbitrage_env import StorageArbitrageEnv
    from baselines.random_policy    import RandomPolicy,    run_episodes
    from baselines.threshold_policy import ThresholdPolicy

    parser = argparse.ArgumentParser(description="Task 4 visualisations")
    parser.add_argument("--n-batteries", type=int, default=1)
    parser.add_argument("--policy",      type=str, default="threshold",
                        choices=["random", "threshold"])
    parser.add_argument("--save",        type=str, default=None,
                        help="Save figure to this path (e.g. episode.png)")
    args = parser.parse_args()

    env = StorageArbitrageEnv(n_batteries=args.n_batteries)

    if args.policy == "random":
        policy = RandomPolicy(env, seed=0)
        label  = "Random Policy"
    else:
        tp = ThresholdPolicy()
        class _W:
            def __init__(self, p, e): self._p = p; self._e = e
            def reset(self):          self._p.reset(env=self._e)
            def act(self, obs):       return self._p.act(obs)
        policy = _W(tp, env)
        label  = "Threshold Policy"

    print(f"Running 1 episode with {label} ...")
    res = run_episodes(policy, env, n_episodes=1, seed=42)
    print(f"Episode profit: ${res['profits'][0]:.2f}")

    plot_episode(res, title=f"{label} — N={args.n_batteries} battery",
                 save_path=args.save)