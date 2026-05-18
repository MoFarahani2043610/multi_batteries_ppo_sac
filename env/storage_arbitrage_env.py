"""
env/storage_arbitrage_env.py
============================
Gymnasium-compatible multi-battery energy storage arbitrage environment.

Implements the MDP from Section 3:
  State   : s_t = [R_t, p_t, f_t]
  Action  : a_t ∈ R^N  (positive = charge, negative = discharge)
  Transition: R_{t+1,i} = clip(R_{t,i} + η_i · a_{t,i} · Δt, 0, R_max,i)
  Reward  : r_t = Σ_i [ -p_{t,k(i)} · a_{t,i} · Δt - λ · |a_{t,i}| ]

Usage
-----
    from env.storage_arbitrage_env import StorageArbitrageEnv
    import numpy as np

    env = StorageArbitrageEnv(n_batteries=3)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Battery configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BatteryConfig:
    """Physical parameters for a single battery unit.

    All power values are in MW, capacity in MWh, time in hours.

    Parameters
    ----------
    capacity_mwh : float
        Maximum energy that can be stored (R_max,i). Default 1.0 MWh.
    p_charge_max : float
        Maximum charging power in MW (positive bound on a_t,i). Default 0.5 MW.
    p_discharge_max : float
        Maximum discharging power in MW (positive, action is negated). Default 0.5 MW.
    eta_charge : float
        One-way charging efficiency ∈ (0, 1]. Default 0.95.
    eta_discharge : float
        One-way discharging efficiency ∈ (0, 1]. Default 0.95.
    soc_min : float
        Minimum allowed state of charge fraction ∈ [0, 1). Default 0.0.
    soc_max : float
        Maximum allowed state of charge fraction ∈ (0, 1]. Default 1.0.
    initial_soc : float or None
        Initial SoC fraction. None → randomised in reset(). Default None.
    market_node : int
        Index into the price vector p_t (k(i) in the formulation). Default 0.
    """
    capacity_mwh: float = 1.0
    p_charge_max: float = 0.5
    p_discharge_max: float = 0.5
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_min: float = 0.0
    soc_max: float = 1.0
    initial_soc: Optional[float] = None
    market_node: int = 0

    def __post_init__(self):
        assert 0 < self.capacity_mwh, "capacity_mwh must be positive"
        assert 0 < self.p_charge_max, "p_charge_max must be positive"
        assert 0 < self.p_discharge_max, "p_discharge_max must be positive"
        assert 0 < self.eta_charge <= 1.0, "eta_charge must be in (0, 1]"
        assert 0 < self.eta_discharge <= 1.0, "eta_discharge must be in (0, 1]"
        assert 0.0 <= self.soc_min < self.soc_max <= 1.0, "invalid SoC bounds"
        if self.initial_soc is not None:
            assert self.soc_min <= self.initial_soc <= self.soc_max, \
                "initial_soc outside [soc_min, soc_max]"

    @property
    def r_min(self) -> float:
        """Absolute minimum SoC in MWh."""
        return self.soc_min * self.capacity_mwh

    @property
    def r_max(self) -> float:
        """Absolute maximum SoC in MWh."""
        return self.soc_max * self.capacity_mwh


def make_homogeneous_fleet(
    n: int,
    capacity_mwh: float = 1.0,
    p_charge_max: float = 0.5,
    p_discharge_max: float = 0.5,
    eta_charge: float = 0.95,
    eta_discharge: float = 0.95,
) -> list[BatteryConfig]:
    """Convenience: N identical batteries all at market node 0."""
    return [
        BatteryConfig(
            capacity_mwh=capacity_mwh,
            p_charge_max=p_charge_max,
            p_discharge_max=p_discharge_max,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Price source protocol
# ---------------------------------------------------------------------------

class PriceSource:
    """Base class for price data providers.

    Subclasses implement ``reset()`` and ``step()`` to feed price vectors
    p_t ∈ R^M and optional feature vectors f_t into the environment.
    """

    @property
    def n_nodes(self) -> int:
        raise NotImplementedError

    @property
    def feature_dim(self) -> int:
        """Dimension of f_t. Return 0 if no features."""
        return 0

    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_0, f_0) at episode start."""
        raise NotImplementedError

    def step(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Advance one timestep. Return (p_t, f_t, done)."""
        raise NotImplementedError


class SyntheticPriceSource(PriceSource):
    """Synthetic single-node price source for unit-testing and quick experiments.

    Generates prices from a mean-reverting process with additive noise and
    configurable daily/weekly seasonality spikes. Fully deterministic given seed.

    Parameters
    ----------
    n_steps : int
        Episode length in timesteps. Default 288 (24 h at 5-min resolution).
    dt_hours : float
        Timestep duration in hours. Default 5/60.
    mean_price : float
        Long-run mean price ($/MWh). Default 30.
    volatility : float
        Noise standard deviation per step. Default 5.
    mean_reversion : float
        Speed of mean reversion ∈ [0, 1]. Default 0.05.
    price_floor : float
        Hard lower bound on price (real markets can go negative). Default -10.
    price_cap : float
        Hard upper bound. Default 500.
    spike_prob : float
        Probability of a price spike per step. Default 0.01.
    spike_magnitude : float
        Multiplicative spike size. Default 8.
    seed : int or None
        Master seed for reproducibility.
    """

    def __init__(
        self,
        n_steps: int = 288,          
        dt_hours: float = 5 / 60,
        mean_price: float = 30.0,
        volatility: float = 5.0,
        mean_reversion: float = 0.05,
        price_floor: float = -10.0,
        price_cap: float = 500.0,
        spike_prob: float = 0.01,
        spike_magnitude: float = 8.0,
        seed: Optional[int] = None,
    ):
        self._n_steps = n_steps
        self._dt = dt_hours
        self._mean = mean_price
        self._vol = volatility
        self._kappa = mean_reversion
        self._floor = price_floor
        self._cap = price_cap
        self._spike_prob = spike_prob
        self._spike_mag = spike_magnitude
        self._master_seed = seed

        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._price: float = mean_price
        self._step_idx: int = 0

    @property
    def n_nodes(self) -> int:
        return 1

    @property
    def feature_dim(self) -> int:
        # f_t = [sin(hour), cos(hour)]  — encodes time of day
        return 2

    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._price = self._mean
        self._step_idx = 0
        return self._observe()

    def step(self) -> tuple[np.ndarray, np.ndarray, bool]:
        # Ornstein-Uhlenbeck mean reversion
        noise = self._rng.normal(0, self._vol)
        self._price += self._kappa * (self._mean - self._price) + noise

        # Random price spike
        if self._rng.random() < self._spike_prob:
            self._price *= self._spike_mag

        self._price = float(np.clip(self._price, self._floor, self._cap))
        self._step_idx += 1
        p, f = self._observe()
        done = self._step_idx >= self._n_steps
        return p, f, done

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        p = np.array([self._price], dtype=np.float32)
        # Circular time encoding: step within a 288-step day
        angle = 2 * np.pi * (self._step_idx % 288) / 288
        f = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
        return p, f


class HistoricalPriceSource(PriceSource):
    """Wraps a pre-loaded price array (e.g. from data/loader.py).

    Parameters
    ----------
    prices : np.ndarray, shape (T, M)
        Price matrix. T timesteps, M market nodes.
    features : np.ndarray or None, shape (T, D)
        Optional feature matrix (forecasts, weather, etc.).
    episode_len : int or None
        If set, sample a random starting index each episode so that
        len(episode) == episode_len. None → use the full array.
    """

    def __init__(
        self,
        prices: np.ndarray,
        features: Optional[np.ndarray] = None,
        episode_len: Optional[int] = None,
    ):
        assert prices.ndim == 2, "prices must be shape (T, M)"
        self._prices = prices.astype(np.float32)
        self._features = features.astype(np.float32) if features is not None else None
        self._T, self._M = prices.shape
        self._D = features.shape[1] if features is not None else 0
        self._episode_len = episode_len or self._T
        self._rng = np.random.default_rng()
        self._start = 0
        self._t = 0

    @property
    def n_nodes(self) -> int:
        return self._M

    @property
    def feature_dim(self) -> int:
        return self._D

    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = self._T - self._episode_len
        self._start = int(self._rng.integers(0, max(1, max_start)))
        self._t = 0
        return self._observe()

    def step(self) -> tuple[np.ndarray, np.ndarray, bool]:
        self._t += 1
        done = self._t >= self._episode_len
        return self._observe() + (done,)

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        idx = self._start + self._t
        p = self._prices[idx]
        f = self._features[idx] if self._features is not None else np.zeros(0, dtype=np.float32)
        return p, f


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------

class StorageArbitrageEnv(gym.Env):
    """Multi-battery energy storage arbitrage environment.

    Implements the MDP from Section 3 of the thesis. Compatible with
    Gymnasium ≥ 0.26 (reset returns (obs, info), step returns 5-tuple).

    Parameters
    ----------
    n_batteries : int
        Number of batteries N. Ignored if ``batteries`` is provided.
    batteries : list[BatteryConfig] or None
        Explicit per-battery configuration. If None, a homogeneous fleet of
        ``n_batteries`` default batteries is created.
    price_source : PriceSource or None
        Price/feature data provider. Defaults to SyntheticPriceSource().
    dt_hours : float
        Timestep duration Δt in hours. Default 5/60 (5 minutes).
    degradation_penalty : float
        λ in the reward: penalty per MW of action magnitude. Default 0.0.
    normalize_obs : bool
        If True, divide SoC by capacity and prices by a reference value.
        Helps neural-network training. Default True.
    price_ref : float
        Reference price for normalisation (e.g. mean LMP). Default 50.0.
    render_mode : str or None
        Currently supports None only (no rendering).

    Observation space
    -----------------
    Box of shape (N + M + D,):
        [R_t/R_max (N),  p_t/price_ref (M),  f_t (D)]

    Action space
    ------------
    Box of shape (N,):
        a_{t,i} ∈ [-p_discharge_max_i, p_charge_max_i]  for each battery i.
        Positive = charge (buy power), negative = discharge (sell power).

    Reward
    ------
    r_t = Σ_i [ -p_{t,k(i)} · a_{t,i} · Δt - λ · |a_{t,i}| ]

    Sign convention: charging costs money (negative reward at positive prices),
    discharging earns money (positive reward at positive prices). The agent
    learns buy-low / sell-high automatically.

    Info dict keys (returned by step and reset)
    -------------------------------------------
    soc_mwh          : np.ndarray (N,)  — absolute SoC in MWh
    prices           : np.ndarray (M,)  — raw prices in $/MWh
    cash_flow        : float            — revenue component of reward
    degradation_cost : float            — penalty component of reward
    actions_clipped  : bool             — whether any action was clipped
    step_idx         : int              — current timestep within episode
    cumulative_profit: float            — total profit so far this episode
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_batteries: int = 1,
        batteries: Optional[list[BatteryConfig]] = None,
        price_source: Optional[PriceSource] = None,
        dt_hours: float = 5 / 60,
        degradation_penalty: float = 0.0,
        normalize_obs: bool = True,
        price_ref: float = 50.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # --- battery fleet ---
        if batteries is not None:
            self.batteries = batteries
        else:
            self.batteries = make_homogeneous_fleet(n_batteries)
        self.N = len(self.batteries)

        # --- price source ---
        self.price_source = price_source or SyntheticPriceSource()
        self.M = self.price_source.n_nodes
        self.D = self.price_source.feature_dim

        # --- parameters ---
        self.dt = dt_hours
        self.lam = degradation_penalty
        self.normalize_obs = normalize_obs
        self.price_ref = price_ref

        # validate market node indices
        for i, b in enumerate(self.batteries):
            assert b.market_node < self.M, (
                f"Battery {i} references market_node={b.market_node} "
                f"but price_source only has {self.M} node(s)."
            )

        # --- spaces ---
        # action: one continuous value per battery
        act_low = np.array([-b.p_discharge_max for b in self.batteries], dtype=np.float32)
        act_high = np.array([b.p_charge_max for b in self.batteries], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # observation: [SoC (N), prices (M), features (D)]
        obs_dim = self.N + self.M + self.D
        obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        if normalize_obs:
            obs_low[:self.N] = 0.0
            obs_high[:self.N] = 1.0          # normalised SoC ∈ [0, 1]
        else:
            obs_low[:self.N] = 0.0
            obs_high[:self.N] = np.array([b.r_max for b in self.batteries])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # --- internal state ---
        self._soc = np.zeros(self.N, dtype=np.float64)
        self._prices = np.zeros(self.M, dtype=np.float64)
        self._features = np.zeros(self.D, dtype=np.float64)
        self._step_idx = 0
        self._cumulative_profit = 0.0
        self._episode_done = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # reset price source
        p0, f0 = self.price_source.reset(seed=seed)
        self._prices = p0.astype(np.float64)
        self._features = f0.astype(np.float64) if len(f0) else np.zeros(self.D)

        # initialise SoC for each battery
        for i, b in enumerate(self.batteries):
            if b.initial_soc is not None:
                self._soc[i] = b.initial_soc * b.capacity_mwh
            else:
                # random uniform in [soc_min, soc_max]
                lo = b.soc_min * b.capacity_mwh
                hi = b.soc_max * b.capacity_mwh
                self._soc[i] = self.np_random.uniform(lo, hi)

        self._step_idx = 0
        self._cumulative_profit = 0.0
        self._episode_done = False

        obs = self._get_obs()
        info = self._get_info(
            actions=np.zeros(self.N),
            cash_flow=0.0,
            degradation_cost=0.0,
            clipped=False,
        )
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert not self._episode_done, (
            "Episode is done. Call reset() before stepping again."
        )

        action = np.asarray(action, dtype=np.float64)

        # --- clip action to feasible power limits ---
        act_low = np.array([-b.p_discharge_max for b in self.batteries])
        act_high = np.array([b.p_charge_max for b in self.batteries])
        action_clipped = np.clip(action, act_low, act_high)
        was_clipped = not np.allclose(action, action_clipped)

        # --- compute reward and update SoC ---
        cash_flow = 0.0
        degradation_cost = 0.0

        for i, b in enumerate(self.batteries):
            a_i = action_clipped[i]
            p_i = self._prices[b.market_node]

            # energy injected into / drawn from battery (MWh)
            if a_i >= 0:
                # charging: grid → battery, efficiency loss on the way in
                delta_energy = b.eta_charge * a_i * self.dt
            else:
                # discharging: battery → grid, efficiency loss on the way out
                delta_energy = (1.0 / b.eta_discharge) * a_i * self.dt  # negative

            # update SoC with physical clip
            new_soc = np.clip(
                self._soc[i] + delta_energy,
                b.r_min,
                b.r_max,
            )

            # actual power delivered to/from grid (may differ after SoC clip)
            if a_i >= 0:
                actual_energy = (new_soc - self._soc[i]) / b.eta_charge
            else:
                actual_energy = (new_soc - self._soc[i]) * b.eta_discharge

            self._soc[i] = new_soc

            # cash flow: negative when buying, positive when selling
            # r = -p · a · Δt  (charging costs money, discharging earns)
            cf_i = -p_i * actual_energy  # actual_energy already has sign
            cash_flow += cf_i

            # degradation penalty
            degradation_cost += self.lam * abs(a_i)

        reward = cash_flow - degradation_cost
        self._cumulative_profit += cash_flow  # track gross profit separately

        # --- advance price source ---
        p_next, f_next, price_done = self.price_source.step()
        self._prices = p_next.astype(np.float64)
        self._features = f_next.astype(np.float64) if len(f_next) else self._features
        self._step_idx += 1

        terminated = price_done
        truncated = False
        self._episode_done = terminated or truncated

        obs = self._get_obs()
        info = self._get_info(
            actions=action_clipped,
            cash_flow=cash_flow,
            degradation_cost=degradation_cost,
            clipped=was_clipped,
        )
        return obs, float(reward), terminated, truncated, info

    def render(self):
        # For full rendering support, use notebooks/01_env_walkthrough.ipynb
        warnings.warn("render() is not implemented. Use the walkthrough notebook.")

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self.normalize_obs:
            soc_obs = np.array(
                [self._soc[i] / self.batteries[i].capacity_mwh for i in range(self.N)],
                dtype=np.float32,
            )
            price_obs = (self._prices / self.price_ref).astype(np.float32)
        else:
            soc_obs = self._soc.astype(np.float32)
            price_obs = self._prices.astype(np.float32)

        feature_obs = self._features.astype(np.float32)
        return np.concatenate([soc_obs, price_obs, feature_obs])

    def _get_info(
        self,
        actions: np.ndarray,
        cash_flow: float,
        degradation_cost: float,
        clipped: bool,
    ) -> dict[str, Any]:
        return {
            "soc_mwh": self._soc.copy(),
            "prices": self._prices.copy(),
            "cash_flow": cash_flow,
            "degradation_cost": degradation_cost,
            "actions_clipped": clipped,
            "step_idx": self._step_idx,
            "cumulative_profit": self._cumulative_profit,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_batteries(self) -> int:
        return self.N

    @property
    def obs_dim(self) -> int:
        return self.N + self.M + self.D

    def __repr__(self) -> str:
        return (
            f"StorageArbitrageEnv("
            f"N={self.N}, M={self.M}, D={self.D}, "
            f"dt={self.dt*60:.0f}min, λ={self.lam})"
        )


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

def _make_env(**kwargs):
    return StorageArbitrageEnv(**kwargs)


# Register so users can do: gym.make("StorageArbitrage-v0")
try:
    gym.register(
        id="StorageArbitrage-v0",
        entry_point="env.storage_arbitrage_env:StorageArbitrageEnv",
        kwargs={},
    )
except Exception:
    pass  # already registered in the same process


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    print("=" * 60)
    print("StorageArbitrageEnv — quick sanity check")
    print("=" * 60)

    # 1. Single battery, synthetic prices
    env = StorageArbitrageEnv(n_batteries=1, degradation_penalty=0.05)
    print(f"\nEnvironment: {env}")
    print(f"  obs_space : {env.observation_space}")
    print(f"  act_space : {env.action_space}")

    obs, info = env.reset(seed=0)
    print(f"\nInitial obs  : {obs}")
    print(f"Initial SoC  : {info['soc_mwh']} MWh")
    print(f"Initial price: {info['prices']} $/MWh")

    total_reward = 0.0
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        if term or trunc:
            break

    print(f"\nAfter 10 steps:")
    print(f"  cumulative profit : ${info['cumulative_profit']:.4f}")
    print(f"  total reward      : ${total_reward:.4f}")
    print(f"  final SoC         : {info['soc_mwh']} MWh")

    # 2. Heterogeneous fleet, N=3
    print("\n" + "-" * 40)
    fleet = [
        BatteryConfig(capacity_mwh=2.0, p_charge_max=1.0, p_discharge_max=1.0,
                      eta_charge=0.95, eta_discharge=0.95, market_node=0),
        BatteryConfig(capacity_mwh=0.5, p_charge_max=0.25, p_discharge_max=0.25,
                      eta_charge=0.90, eta_discharge=0.90, market_node=0),
        BatteryConfig(capacity_mwh=1.0, p_charge_max=0.5, p_discharge_max=0.5,
                      eta_charge=0.98, eta_discharge=0.98, market_node=0),
    ]
    env3 = StorageArbitrageEnv(batteries=fleet)
    print(f"Heterogeneous fleet: {env3}")
    obs3, _ = env3.reset(seed=42)
    print(f"  obs shape: {obs3.shape}  (expected: {env3.obs_dim})")
    print(f"  act shape: {env3.action_space.shape}  (expected: (3,))")

    # 3. Reproducibility check
    print("\n" + "-" * 40)
    env_a = StorageArbitrageEnv(n_batteries=2)
    env_b = StorageArbitrageEnv(n_batteries=2)
    obs_a, _ = env_a.reset(seed=99)
    obs_b, _ = env_b.reset(seed=99)
    assert np.allclose(obs_a, obs_b), "Reproducibility FAILED"
    print("Reproducibility check: PASSED (same seed → same obs)")
    print("=" * 60)