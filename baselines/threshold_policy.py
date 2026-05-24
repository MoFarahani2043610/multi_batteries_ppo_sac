"""
baselines/threshold_policy.py
==============================
Threshold policy — simple rule-based baseline, no learning.

Rule
----
  price ≤ charge_threshold    → charge all batteries at full power  (buy cheap)
  price ≥ discharge_threshold → discharge all batteries at full power (sell dear)
  otherwise                   → idle (zero action)

Two variants
------------
  ThresholdPolicy         — fixed thresholds set at construction time
  AdaptiveThresholdPolicy — thresholds update from a rolling price window

Purpose
-------
Sits between random (floor) and LP (ceiling).
A trained DRL agent should beat this; if it does not, training is failing.

Run
---
    python baselines/threshold_policy.py --n-batteries 3 --n-episodes 20
    python baselines/threshold_policy.py --adaptive
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from baselines.random_policy import run_episodes


class ThresholdPolicy:
    """
    Fixed charge/discharge price thresholds.

    Parameters
    ----------
    charge_threshold    : charge when price ≤ this ($/MWh)
    discharge_threshold : discharge when price ≥ this ($/MWh)
    """

    name = "Threshold"

    def __init__(self, charge_threshold=25.0, discharge_threshold=50.0):
        self.charge_thr    = charge_threshold
        self.discharge_thr = discharge_threshold
        self._env          = None

    def reset(self, env=None):
        if env is not None:
            self._env = env

    def act(self, obs):
        """
        Read price from obs[N] (price starts after the N SoC entries).
        Return full charge, full discharge, or zero.
        """
        assert self._env is not None, "Call reset(env) before act()."
        N     = self._env.N
        # obs prices are normalised by price_ref — denormalise for comparison
        price = float(obs[N]) * self._env.price_ref

        if price <= self.charge_thr:
            # buy cheap — charge all batteries at full power
            return np.array(
                [b.p_charge_max for b in self._env.batteries],
                dtype=np.float32,
            )
        elif price >= self.discharge_thr:
            # sell dear — discharge all batteries at full power
            return np.array(
                [-b.p_discharge_max for b in self._env.batteries],
                dtype=np.float32,
            )
        else:
            return np.zeros(N, dtype=np.float32)


class AdaptiveThresholdPolicy(ThresholdPolicy):
    """
    Thresholds adapt from a rolling window of recent prices.

    Every timestep, keeps the last `window` prices and sets:
      charge_threshold    = q_low  percentile of window
      discharge_threshold = q_high percentile of window

    Smarter than fixed thresholds but still entirely rule-based.

    Parameters
    ----------
    window  : int   — rolling window in timesteps (default 288 = 1 day)
    q_low   : float — charge below this percentile  (default 25)
    q_high  : float — discharge above this percentile (default 75)
    """

    name = "Adaptive Threshold"

    def __init__(self, window=288, q_low=25.0, q_high=75.0):
        super().__init__(charge_threshold=25.0, discharge_threshold=50.0)
        self.window  = window
        self.q_low   = q_low
        self.q_high  = q_high
        self._history = []

    def reset(self, env=None):
        super().reset(env)
        self._history = []

    def act(self, obs):
        assert self._env is not None, "Call reset(env) before act()."
        price = float(obs[self._env.N]) * self._env.price_ref
        self._history.append(price)

        # update thresholds once window is full
        if len(self._history) >= self.window:
            w = self._history[-self.window:]
            self.charge_thr    = float(np.percentile(w, self.q_low))
            self.discharge_thr = float(np.percentile(w, self.q_high))

        return super().act(obs)


# ─────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from storage_arbitrage_env import StorageArbitrageEnv

    parser = argparse.ArgumentParser(description="Threshold policy baseline")
    parser.add_argument("--n-batteries",   type=int,   default=1)
    parser.add_argument("--n-episodes",    type=int,   default=10)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--charge-thr",    type=float, default=25.0)
    parser.add_argument("--discharge-thr", type=float, default=50.0)
    parser.add_argument("--adaptive",      action="store_true")
    args = parser.parse_args()

    env = StorageArbitrageEnv(n_batteries=args.n_batteries)

    if args.adaptive:
        policy = AdaptiveThresholdPolicy()
        label  = "Adaptive Threshold Policy"
    else:
        policy = ThresholdPolicy(args.charge_thr, args.discharge_thr)
        label  = (f"Threshold Policy "
                  f"(charge≤{args.charge_thr}, discharge≥{args.discharge_thr})")

    # threshold policies need env attached at reset
    class _Wrapped:
        def __init__(self, p, e):
            self._p = p; self._e = e
        def reset(self):        self._p.reset(env=self._e)
        def act(self, obs):     return self._p.act(obs)

    res = run_episodes(_Wrapped(policy, env), env,
                       n_episodes=args.n_episodes, seed=args.seed)

    print(f"\n{label}")
    print(f"  N={args.n_batteries} batteries, {args.n_episodes} episodes")
    print(f"  Mean profit : ${res['mean_profit']:>8.4f}")
    print(f"  Std         : ${res['std_profit']:>8.4f}")
    print(f"  Min / Max   : ${res['min_profit']:.4f} / ${res['max_profit']:.4f}")
    print(f"\n  → DRL must beat this to be considered useful.")