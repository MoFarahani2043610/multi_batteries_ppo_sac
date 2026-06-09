import sys, os
sys.path.insert(0, r'D:\Project 2026\env\env')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
baselines/perfect_foresight.py
================================
Perfect-foresight LP solver — the theoretical upper bound on profit.

This policy CHEATS: it receives ALL future prices before the episode
starts and solves the exact optimal schedule with cvxpy.
No real agent can beat it — it is the ceiling.

The "optimality gap" = (LP_profit − DRL_profit) / LP_profit
tells you how close your trained agent is to the theoretical best.

Install
-------
    pip install cvxpy

LP formulation
--------------
For each battery i and timestep t:

  Maximise:
    Σ_t Σ_i  [ p_{t,k(i)} · discharge_{t,i} · Δt      ← revenue from selling
              − p_{t,k(i)} · charge_{t,i}    · Δt      ← cost of buying
              − λ · (charge_{t,i} + discharge_{t,i}) ]  ← degradation penalty

  Subject to:
    SoC_{t+1,i} = SoC_{t,i}
                + η_chg,i  · charge_{t,i}    · Δt      ← energy stored
                − (1/η_dis,i) · discharge_{t,i} · Δt   ← energy released
    0 ≤ SoC_{t,i}        ≤ R_max,i
    0 ≤ charge_{t,i}     ≤ P_chg_max,i
    0 ≤ discharge_{t,i}  ≤ P_dis_max,i

Note: action a_{t,i} = charge_{t,i} − discharge_{t,i}
We split into two non-negative variables because LP requires that.

Run
---
    pip install cvxpy
    python baselines/perfect_foresight.py --n-batteries 1 --n-episodes 5
    python baselines/perfect_foresight.py --compare   ← all three baselines
"""

import sys, os

from storage_arbitrage_env import StorageArbitrageEnv

sys.path.insert(0, r"D:\Project 2026\env\env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
try:
    from baselines.random_policy import run_episodes
except ModuleNotFoundError:
    from random_policy import run_episodes

# ─────────────────────────────────────────────────────
#  Helper: extract full episode price sequence
# ─────────────────────────────────────────────────────

def _get_episode_prices(env):
    import copy
    src = getattr(env, 'price_source', None) or getattr(env, 'prices', None)
    if src is None:
        raise AttributeError("Cannot find price source.")

    # HistoricalPrices
    if hasattr(src, "_prices") and hasattr(src, "_start"):
        ep_len = getattr(src, 'episode_len', getattr(src, '_n_steps', 288))
        return src._prices[src._start:src._start + ep_len].astype(np.float64)

    # SyntheticPriceSource (any variant)
    elif hasattr(src, "_price"):
        T           = getattr(src, '_n_steps', getattr(src, 'episode_steps', 288))
        saved_price = src._price
        saved_t     = getattr(src, '_step_idx', getattr(src, '_t', 0))
        saved_rng   = copy.deepcopy(src._rng)
        prices_list = [src._price]
        for _ in range(T - 1):
            p, _, done = src.step()
            prices_list.append(float(p[0]))
            if done:
                break
        src._price = saved_price
        if hasattr(src, '_step_idx'):
            src._step_idx = saved_t
        else:
            src._t = saved_t
        src._rng = saved_rng
        return np.array(prices_list, dtype=np.float64).reshape(-1, 1)

    else:
        raise ValueError("Use SyntheticPrices or HistoricalPrices.")

# ─────────────────────────────────────────────────────
#  LP solver
# ─────────────────────────────────────────────────────

class PerfectForesightLP:
    """
    Solves the exact multi-battery arbitrage LP given full future prices.

    Parameters
    ----------
    env     : StorageArbitrageEnv
    solver  : cvxpy solver — "CLARABEL" (default, ships with cvxpy),
              "ECOS", "SCS", or "GLPK"
    verbose : print solver output (default False)
    """

    name = "Perfect Foresight LP"

    def __init__(self, env, solver="CLARABEL", verbose=False):
        self.env     = env
        self.solver  = solver
        self.verbose = verbose

    def solve(self, prices, initial_soc):
        """
        Solve the LP for one episode.

        Parameters
        ----------
        prices      : np.ndarray (T, M) — full price sequence (the cheat)
        initial_soc : np.ndarray (N,)   — starting SoC in MWh

        Returns
        -------
        dict
            profit       : gross profit ($) — revenue minus buy cost
            lp_value     : LP objective value (includes λ penalty)
            charge       : (T, N) charge schedule (MW)
            discharge    : (T, N) discharge schedule (MW)
            action       : (T, N) = charge − discharge
            soc          : (T+1, N) SoC trajectory (MWh)
            solve_time_s : solver wall-clock time (seconds)
            status       : solver status string
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError(
                "cvxpy not installed.\n"
                "Run:  pip install cvxpy"
            )
        import time

        prices = np.asarray(prices, dtype=np.float64)
        T      = prices.shape[0]
        N      = self.env.N
        dt     = self.env.dt
        lam    = self.env.lam
        bats   = self.env.batteries

        # ── decision variables ───────────────────────────────────────
        chg = cp.Variable((T, N), nonneg=True, name="charge")      # MW ≥ 0
        dis = cp.Variable((T, N), nonneg=True, name="discharge")    # MW ≥ 0
        soc = cp.Variable((T+1, N), nonneg=True, name="soc")        # MWh ≥ 0

        # ── objective ────────────────────────────────────────────────
        obj = 0.0
        for i, b in enumerate(bats):
            p = prices[:, b.market_node]          # (T,) price at this node
            obj += cp.sum(
                  cp.multiply(p,  dis[:, i]) * dt  # revenue: sell → earn
                - cp.multiply(p,  chg[:, i]) * dt  # cost:    buy  → spend
                - lam * (chg[:, i] + dis[:, i])    # degradation penalty
            )

        # ── constraints ──────────────────────────────────────────────
        cons = []

        # initial SoC
        for i in range(N):
            cons.append(soc[0, i] == float(initial_soc[i]))

        for t in range(T):
            for i, b in enumerate(bats):
                # SoC dynamics — matches env transition exactly
                cons.append(
                    soc[t+1, i] == soc[t, i]
                        + b.eta_charge           * chg[t, i] * dt
                        - (1.0 / b.eta_discharge)* dis[t, i] * dt
                )
                # SoC bounds
                cons.append(soc[t+1, i] <= b.capacity_mwh)

                # power limits
                cons.append(chg[t, i] <= b.p_charge_max)
                cons.append(dis[t, i] <= b.p_discharge_max)

        # ── solve ─────────────────────────────────────────────────────
        prob = cp.Problem(cp.Maximize(obj), cons)
        t0   = time.perf_counter()
        prob.solve(solver=self.solver, verbose=self.verbose)
        elapsed = time.perf_counter() - t0

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"LP solver status: '{prob.status}'. "
                "Try solver='ECOS' or solver='SCS'."
            )

        chg_val = np.maximum(chg.value, 0.0)
        dis_val = np.maximum(dis.value, 0.0)
        soc_val = soc.value

        # gross profit (without λ penalty, for fair comparison with DRL)
        profit = 0.0
        for i, b in enumerate(bats):
            p = prices[:, b.market_node]
            profit += float(
                np.sum(p * dis_val[:, i] * dt)
              - np.sum(p * chg_val[:, i] * dt)
            )

        return {
            "profit":       profit,
            "lp_value":     float(prob.value),
            "charge":       chg_val,
            "discharge":    dis_val,
            "action":       chg_val - dis_val,
            "soc":          soc_val,
            "solve_time_s": elapsed,
            "status":       prob.status,
        }


# ─────────────────────────────────────────────────────
#  Episode-level evaluation
# ─────────────────────────────────────────────────────

def evaluate_lp(env, n_episodes=5, seed=0, solver="CLARABEL", drl_profits=None):
    """
    Run the LP across n_episodes and report profit + optimality gap.

    Parameters
    ----------
    env          : StorageArbitrageEnv
    n_episodes   : int
    seed         : int
    solver       : cvxpy solver name
    drl_profits  : optional list of DRL profits for gap calculation

    Returns
    -------
    dict: profits, mean_profit, std_profit, solve_times,
          optimality_gap (if drl_profits provided)
          all_soc, all_prices, all_actions  (last episode, for plotting)
    """
    lp      = PerfectForesightLP(env, solver=solver)
    profits = []
    times   = []

    last_soc = last_prices = last_actions = None

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)

        prices_ep = _get_episode_prices(env)     # (T, M) — the cheat
        result    = lp.solve(
            prices      = prices_ep,
            initial_soc = info["soc_mwh"],
        )

        profits.append(result["profit"])
        times.append(result["solve_time_s"])

        print(f"  Episode {ep+1}/{n_episodes}:  "
              f"LP profit = ${result['profit']:>8.4f}  "
              f"({result['solve_time_s']:.2f}s, {result['status']})")

        if ep == n_episodes - 1:
            T = prices_ep.shape[0]
            last_soc     = result["soc"]                          # (T+1, N)
            last_prices  = [prices_ep[t] for t in range(T)]       # list of (M,)
            last_actions = [result["action"][t] for t in range(T)]# list of (N,)

    profits = np.array(profits)
    out = {
        "profits":          profits.tolist(),
        "mean_profit":      float(profits.mean()),
        "std_profit":       float(profits.std()),
        "min_profit":       float(profits.min()),
        "max_profit":       float(profits.max()),
        "solve_times":      times,
        "mean_solve_time":  float(np.mean(times)),
        "optimality_gap":   None,
        "all_soc":          last_soc,
        "all_prices":       last_prices,
        "all_actions":      last_actions,
    }

    if drl_profits is not None:
        n  = min(len(drl_profits), len(profits))
        lp_a  = profits[:n]
        drl_a = np.array(drl_profits[:n])
        gaps  = (lp_a - drl_a) / np.maximum(np.abs(lp_a), 1e-6)
        out["optimality_gap"] = float(gaps.mean())
        print(f"\n  Mean optimality gap vs DRL: {out['optimality_gap']:.1%}")

    return out


# ─────────────────────────────────────────────────────
#  Compare all three baselines in one call
# ─────────────────────────────────────────────────────

def compare_all(env, n_episodes=5, seed=0):
    """
    Run random, threshold, and LP policies and print a comparison table.

    Expected ordering:
        random profit  <  threshold profit  <  LP profit

    Your trained DRL agent should land between threshold and LP.
    """
    try:
        from baselines.random_policy    import RandomPolicy
        from baselines.threshold_policy import ThresholdPolicy
    except ModuleNotFoundError:
        from random_policy    import RandomPolicy
        from threshold_policy import ThresholdPolicy

    print(f"\n{'─'*54}")
    print(f"  Baseline comparison — N={env.N} batteries, "
          f"{n_episodes} episodes")
    print(f"{'─'*54}")

    # ── Random ──────────────────────────────────────────────────
    rp  = RandomPolicy(env, seed=seed)
    r   = run_episodes(rp, env, n_episodes=n_episodes, seed=seed)
    print(f"  Random     (lower bound) : "
          f"${r['mean_profit']:>8.4f}  ±{r['std_profit']:.4f}")

    # ── Threshold ────────────────────────────────────────────────
    tp = ThresholdPolicy()

    class _W:   # wrap so reset() passes env
        def __init__(self, p, e): self._p = p; self._e = e
        def reset(self):      self._p.reset(env=self._e)
        def act(self, obs):   return self._p.act(obs)

    t = run_episodes(_W(tp, env), env, n_episodes=n_episodes, seed=seed)
    print(f"  Threshold  (rule-based)  : "
          f"${t['mean_profit']:>8.4f}  ±{t['std_profit']:.4f}")

    # ── LP ───────────────────────────────────────────────────────
    print(f"  LP         (upper bound) :  solving {n_episodes} episodes …")
    lp = evaluate_lp(env, n_episodes=n_episodes, seed=seed)
    print(f"  LP         (upper bound) : "
          f"${lp['mean_profit']:>8.4f}  ±{lp['std_profit']:.4f}  "
          f"(avg {lp['mean_solve_time']:.2f}s/ep)")

    print(f"{'─'*54}")
    spread = lp['mean_profit'] - r['mean_profit']
    print(f"  Exploitable spread : ${spread:.4f}  (LP − random)")
    print(f"  DRL target zone    : "
          f">${t['mean_profit']:.4f}  →  ${lp['mean_profit']:.4f}")
    print(f"{'─'*54}\n")


# ─────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perfect-foresight LP — theoretical upper bound"
    )
    parser.add_argument("--n-batteries", type=int,  default=1)
    parser.add_argument("--n-episodes",  type=int,  default=5)
    parser.add_argument("--seed",        type=int,  default=0)
    parser.add_argument("--solver",      type=str,  default="CLARABEL",
                        help="cvxpy solver: CLARABEL (default), ECOS, SCS")
    parser.add_argument("--compare",     action="store_true",
                        help="Run all three baselines and print comparison table")
    args = parser.parse_args()

    env = StorageArbitrageEnv(n_batteries=args.n_batteries)

    if args.compare:
        compare_all(env, n_episodes=args.n_episodes, seed=args.seed)

    else:
        print(f"\nPerfect Foresight LP — N={args.n_batteries} batteries, "
              f"{args.n_episodes} episodes")
        res = evaluate_lp(
            env,
            n_episodes = args.n_episodes,
            seed       = args.seed,
            solver     = args.solver,
        )
        print(f"\n  Mean profit  : ${res['mean_profit']:>8.4f}")
        print(f"  Std          : ${res['std_profit']:>8.4f}")
        print(f"  Min / Max    : ${res['min_profit']:.4f} / ${res['max_profit']:.4f}")
        print(f"  Avg solve    : {res['mean_solve_time']:.2f}s per episode")
        print(f"\n  → This is the UPPER BOUND. No real agent can beat this.")

