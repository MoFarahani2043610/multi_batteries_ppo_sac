"""
tests/test_env.py
=================
pytest suite verifying the StorageArbitrageEnv:
  - Physical constraints (SoC never violates bounds)
  - Reward formula correctness
  - Action space and observation space shapes
  - Seed reproducibility
  - Gymnasium API compliance (step/reset signatures)
  - Heterogeneous fleet support
  - Edge cases (full battery, empty battery, zero action)

Run with:
    pytest tests/test_env.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from env.storage_arbitrage_env import (
    BatteryConfig,
    HistoricalPriceSource,
    StorageArbitrageEnv,
    SyntheticPriceSource,
    make_homogeneous_fleet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env1():
    """Single-battery environment, default settings."""
    e = StorageArbitrageEnv(n_batteries=1, degradation_penalty=0.0)
    yield e
    e.close()


@pytest.fixture
def env5():
    """Five-battery homogeneous fleet."""
    e = StorageArbitrageEnv(n_batteries=5, degradation_penalty=0.0)
    yield e
    e.close()


@pytest.fixture
def env_het():
    """Heterogeneous fleet: two batteries with different specs."""
    fleet = [
        BatteryConfig(capacity_mwh=2.0, p_charge_max=1.0, p_discharge_max=1.0,
                      eta_charge=0.95, eta_discharge=0.95, market_node=0),
        BatteryConfig(capacity_mwh=0.5, p_charge_max=0.2, p_discharge_max=0.2,
                      eta_charge=0.85, eta_discharge=0.85, market_node=0),
    ]
    e = StorageArbitrageEnv(batteries=fleet, normalize_obs=False)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. Gymnasium API compliance
# ---------------------------------------------------------------------------

class TestGymnasiumAPI:
    def test_reset_returns_two_tuple(self, env1):
        result = env1.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, env1):
        env1.reset(seed=0)
        result = env1.step(env1.action_space.sample())
        assert isinstance(result, tuple) and len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_in_space(self, env1):
        obs, _ = env1.reset(seed=0)
        assert env1.observation_space.contains(obs), (
            f"Observation {obs} not in space {env1.observation_space}"
        )

    def test_action_space_dtype(self, env1):
        a = env1.action_space.sample()
        assert a.dtype == np.float32

    def test_step_raises_after_done(self, env1):
        """Environment must raise if stepped after episode ends."""
        env1.reset(seed=0)
        # exhaust the episode (SyntheticPriceSource default: 288 steps)
        done = False
        for _ in range(400):
            if done:
                break
            _, _, term, trunc, _ = env1.step(env1.action_space.sample())
            done = term or trunc
        with pytest.raises(AssertionError):
            env1.step(env1.action_space.sample())


# ---------------------------------------------------------------------------
# 2. Physical constraints — SoC must stay in bounds
# ---------------------------------------------------------------------------

class TestPhysicalConstraints:
    def _run_episode(self, env, action_fn, seed=0):
        """Run a full episode, collecting per-step SoC arrays."""
        env.reset(seed=seed)
        socs = []
        done = False
        while not done:
            a = action_fn(env)
            obs, _, term, trunc, info = env.step(a)
            socs.append(info["soc_mwh"].copy())
            done = term or trunc
        return np.array(socs)

    def test_soc_never_negative_random_actions(self, env1):
        socs = self._run_episode(env1, lambda e: e.action_space.sample())
        assert np.all(socs >= -1e-9), f"Negative SoC detected: min={socs.min()}"

    def test_soc_never_exceeds_capacity_random_actions(self, env1):
        socs = self._run_episode(env1, lambda e: e.action_space.sample())
        cap = env1.batteries[0].capacity_mwh
        assert np.all(socs <= cap + 1e-9), f"SoC exceeded capacity: max={socs.max()}"

    def test_soc_never_negative_full_charge(self, env1):
        """Always charging at max power should never give negative SoC."""
        socs = self._run_episode(
            env1, lambda e: np.array([e.batteries[0].p_charge_max], dtype=np.float32)
        )
        assert np.all(socs >= -1e-9)

    def test_soc_never_exceeds_capacity_full_discharge(self, env1):
        """Always discharging at max power should never exceed capacity."""
        socs = self._run_episode(
            env1, lambda e: np.array([-e.batteries[0].p_discharge_max], dtype=np.float32)
        )
        cap = env1.batteries[0].capacity_mwh
        assert np.all(socs <= cap + 1e-9)

    def test_zero_action_soc_unchanged(self):
        """Zero power command — SoC must stay exactly constant."""
        env = StorageArbitrageEnv(
            n_batteries=1,
            batteries=[BatteryConfig(initial_soc=0.5)],
        )
        env.reset(seed=0)
        initial_soc = env._soc.copy()
        for _ in range(5):
            _, _, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
            np.testing.assert_allclose(
                info["soc_mwh"], initial_soc, atol=1e-9,
                err_msg="SoC changed under zero action"
            )
            if term or trunc:
                break
        env.close()

    def test_soc_bounds_heterogeneous(self, env_het):
        """Both batteries in a heterogeneous fleet stay in bounds."""
        env_het.reset(seed=7)
        for _ in range(50):
            _, _, term, trunc, info = env_het.step(env_het.action_space.sample())
            for i, b in enumerate(env_het.batteries):
                assert info["soc_mwh"][i] >= b.r_min - 1e-9
                assert info["soc_mwh"][i] <= b.r_max + 1e-9
            if term or trunc:
                break

    def test_multi_battery_soc_bounds(self, env5):
        env5.reset(seed=3)
        for _ in range(100):
            _, _, term, trunc, info = env5.step(env5.action_space.sample())
            for i, b in enumerate(env5.batteries):
                assert info["soc_mwh"][i] >= b.r_min - 1e-9
                assert info["soc_mwh"][i] <= b.r_max + 1e-9
            if term or trunc:
                break


# ---------------------------------------------------------------------------
# 3. Reward formula correctness
# ---------------------------------------------------------------------------

class TestRewardFormula:
    def test_cash_flow_sign_charging(self):
        """Charging (positive action) at positive price costs money."""
        # Fix price to 40 $/MWh, charge at 0.5 MW for Δt=5min
        prices = np.full((300, 1), 40.0, dtype=np.float32)
        src = HistoricalPriceSource(prices, episode_len=288)
        env = StorageArbitrageEnv(
            batteries=[BatteryConfig(initial_soc=0.0, capacity_mwh=10.0,
                                     p_charge_max=0.5, p_discharge_max=0.5,
                                     eta_charge=1.0, eta_discharge=1.0)],
            price_source=src,
            degradation_penalty=0.0,
            normalize_obs=False,
        )
        env.reset(seed=0)
        # charge at full rate
        _, reward, _, _, info = env.step(np.array([0.5], dtype=np.float32))
        # expected cash flow: -price * power * dt = -40 * 0.5 * (5/60) ≈ -1.6667
        expected = -40.0 * 0.5 * (5 / 60)
        assert reward == pytest.approx(expected, rel=1e-4), (
            f"Charging reward {reward} ≠ expected {expected}"
        )
        env.close()

    def test_cash_flow_sign_discharging(self):
        """Discharging (negative action) at positive price earns money."""
        prices = np.full((300, 1), 40.0, dtype=np.float32)
        src = HistoricalPriceSource(prices, episode_len=288)
        env = StorageArbitrageEnv(
            batteries=[BatteryConfig(initial_soc=1.0, capacity_mwh=10.0,
                                     p_charge_max=0.5, p_discharge_max=0.5,
                                     eta_charge=1.0, eta_discharge=1.0)],
            price_source=src,
            degradation_penalty=0.0,
            normalize_obs=False,
        )
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.array([-0.5], dtype=np.float32))
        # expected: +price * power * dt = +40 * 0.5 * (5/60)
        expected = +40.0 * 0.5 * (5 / 60)
        assert reward == pytest.approx(expected, rel=1e-4)
        env.close()

    def test_degradation_penalty_applied(self):
        """With λ > 0, reward must be strictly less than cash flow."""
        prices = np.full((300, 1), 40.0, dtype=np.float32)
        src = HistoricalPriceSource(prices, episode_len=288)
        env = StorageArbitrageEnv(
            batteries=[BatteryConfig(initial_soc=1.0, capacity_mwh=10.0,
                                     p_charge_max=0.5, p_discharge_max=0.5,
                                     eta_charge=1.0, eta_discharge=1.0)],
            price_source=src,
            degradation_penalty=1.0,
            normalize_obs=False,
        )
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.array([-0.5], dtype=np.float32))
        assert reward < info["cash_flow"], (
            "Degradation penalty not subtracted from reward"
        )
        assert info["degradation_cost"] == pytest.approx(1.0 * 0.5, rel=1e-6)
        env.close()

    def test_zero_action_zero_reward(self):
        """Zero action must always yield zero reward."""
        env = StorageArbitrageEnv(n_batteries=2, degradation_penalty=0.5)
        env.reset(seed=0)
        for _ in range(5):
            _, reward, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
            assert reward == pytest.approx(0.0, abs=1e-9)
            if term or trunc:
                break
        env.close()

    def test_reward_multi_battery_additivity(self):
        """Two identical batteries acting identically → reward = 2× single battery."""
        prices = np.full((300, 1), 30.0, dtype=np.float32)
        kwargs = dict(
            capacity_mwh=10.0, p_charge_max=0.5, p_discharge_max=0.5,
            initial_soc=1.0, eta_charge=1.0, eta_discharge=1.0,
        )
        # single battery
        src1 = HistoricalPriceSource(prices.copy(), episode_len=288)
        env1 = StorageArbitrageEnv(
            batteries=[BatteryConfig(**kwargs)],
            price_source=src1, degradation_penalty=0.0, normalize_obs=False,
        )
        env1.reset(seed=0)
        _, r1, _, _, _ = env1.step(np.array([-0.3], dtype=np.float32))

        # two identical batteries
        src2 = HistoricalPriceSource(prices.copy(), episode_len=288)
        env2 = StorageArbitrageEnv(
            batteries=[BatteryConfig(**kwargs), BatteryConfig(**kwargs)],
            price_source=src2, degradation_penalty=0.0, normalize_obs=False,
        )
        env2.reset(seed=0)
        _, r2, _, _, _ = env2.step(np.array([-0.3, -0.3], dtype=np.float32))

        assert r2 == pytest.approx(2 * r1, rel=1e-5)
        env1.close(); env2.close()


# ---------------------------------------------------------------------------
# 4. Seed reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def _collect_trajectory(self, env, seed, n_steps=20):
        obs_list, reward_list = [], []
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(n_steps):
            a = rng.uniform(
                env.action_space.low, env.action_space.high
            ).astype(np.float32)
            obs, r, term, trunc, _ = env.step(a)
            obs_list.append(obs.copy())
            reward_list.append(r)
            if term or trunc:
                break
        return obs_list, reward_list

    def test_same_seed_same_trajectory(self, env1):
        obs_a, rew_a = self._collect_trajectory(env1, seed=42)
        obs_b, rew_b = self._collect_trajectory(env1, seed=42)
        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_array_equal(a, b, err_msg=f"Obs mismatch at step {i}")
        assert rew_a == rew_b

    def test_different_seed_different_trajectory(self, env1):
        obs_a, _ = self._collect_trajectory(env1, seed=1)
        obs_b, _ = self._collect_trajectory(env1, seed=2)
        # At least one observation should differ
        differ = any(not np.allclose(a, b) for a, b in zip(obs_a, obs_b))
        assert differ, "Different seeds produced identical trajectories"

    def test_reproducibility_multi_battery(self, env5):
        obs_a, rew_a = self._collect_trajectory(env5, seed=7)
        obs_b, rew_b = self._collect_trajectory(env5, seed=7)
        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_array_equal(a, b, err_msg=f"Obs mismatch at step {i}")


# ---------------------------------------------------------------------------
# 5. Observation and action space shapes
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_obs_shape_single(self, env1):
        obs, _ = env1.reset(seed=0)
        # N=1, M=1 (synthetic), D=2 (sin/cos)
        assert obs.shape == (4,), f"Expected (4,) got {obs.shape}"

    def test_obs_shape_multi(self, env5):
        obs, _ = env5.reset(seed=0)
        # N=5, M=1, D=2
        assert obs.shape == (8,), f"Expected (8,) got {obs.shape}"

    def test_action_shape_single(self, env1):
        env1.reset(seed=0)
        assert env1.action_space.shape == (1,)

    def test_action_shape_multi(self, env5):
        env5.reset(seed=0)
        assert env5.action_space.shape == (5,)

    def test_action_bounds_respected(self, env5):
        """Sampled actions must always be within declared bounds."""
        env5.reset(seed=0)
        for _ in range(100):
            a = env5.action_space.sample()
            assert np.all(a >= env5.action_space.low)
            assert np.all(a <= env5.action_space.high)

    def test_obs_normalized_in_01_for_soc(self, env5):
        """With normalize_obs=True, SoC components must lie in [0, 1]."""
        obs, _ = env5.reset(seed=0)
        soc_obs = obs[:env5.N]
        assert np.all(soc_obs >= 0.0) and np.all(soc_obs <= 1.0 + 1e-6)
        for _ in range(20):
            obs, _, term, trunc, _ = env5.step(env5.action_space.sample())
            soc_obs = obs[:env5.N]
            assert np.all(soc_obs >= 0.0 - 1e-9) and np.all(soc_obs <= 1.0 + 1e-6)
            if term or trunc:
                break


# ---------------------------------------------------------------------------
# 6. Historical price source integration
# ---------------------------------------------------------------------------

class TestHistoricalPriceSource:
    def test_episode_length_respected(self):
        prices = np.random.default_rng(0).uniform(20, 80, (1000, 1)).astype(np.float32)
        src = HistoricalPriceSource(prices, episode_len=100)
        env = StorageArbitrageEnv(n_batteries=1, price_source=src)
        env.reset(seed=0)
        step_count = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
            step_count += 1
            done = term or trunc
        assert step_count == 100, f"Expected 100 steps, got {step_count}"
        env.close()

    def test_multi_node_prices(self):
        """Two market nodes, batteries assigned to different nodes."""
        prices = np.column_stack([
            np.full(300, 30.0),   # node 0
            np.full(300, 60.0),   # node 1
        ]).astype(np.float32)
        src = HistoricalPriceSource(prices, episode_len=100)
        fleet = [
            BatteryConfig(market_node=0, initial_soc=1.0,
                          eta_charge=1.0, eta_discharge=1.0),
            BatteryConfig(market_node=1, initial_soc=1.0,
                          eta_charge=1.0, eta_discharge=1.0),
        ]
        env = StorageArbitrageEnv(batteries=fleet, price_source=src,
                                   degradation_penalty=0.0, normalize_obs=False)
        env.reset(seed=0)
        action = np.array([-0.5, -0.5], dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        # Battery 0 discharges at node 0 (price 30), battery 1 at node 1 (price 60)
        # reward = 30*0.5*dt + 60*0.5*dt = (30+60)*0.5*(5/60)
        expected = (30.0 + 60.0) * 0.5 * (5 / 60)
        assert reward == pytest.approx(expected, rel=1e-4)
        env.close()


# ---------------------------------------------------------------------------
# 7. Battery configuration validation
# ---------------------------------------------------------------------------

class TestBatteryConfig:
    def test_invalid_capacity_raises(self):
        with pytest.raises(AssertionError):
            BatteryConfig(capacity_mwh=-1.0)

    def test_invalid_efficiency_raises(self):
        with pytest.raises(AssertionError):
            BatteryConfig(eta_charge=1.5)

    def test_invalid_soc_bounds_raises(self):
        with pytest.raises(AssertionError):
            BatteryConfig(soc_min=0.8, soc_max=0.3)

    def test_invalid_initial_soc_raises(self):
        with pytest.raises(AssertionError):
            BatteryConfig(initial_soc=1.5)

    def test_market_node_validation(self):
        fleet = [BatteryConfig(market_node=5)]
        with pytest.raises(AssertionError):
            # price source only has 1 node (index 0)
            StorageArbitrageEnv(batteries=fleet)


# ---------------------------------------------------------------------------
# 8. Info dict completeness
# ---------------------------------------------------------------------------

class TestInfoDict:
    REQUIRED_KEYS = {
        "soc_mwh", "prices", "cash_flow", "degradation_cost",
        "actions_clipped", "step_idx", "cumulative_profit",
    }

    def test_reset_info_keys(self, env1):
        _, info = env1.reset(seed=0)
        assert self.REQUIRED_KEYS.issubset(info.keys())

    def test_step_info_keys(self, env1):
        env1.reset(seed=0)
        _, _, _, _, info = env1.step(env1.action_space.sample())
        assert self.REQUIRED_KEYS.issubset(info.keys())

    def test_cumulative_profit_accumulates(self, env1):
        env1.reset(seed=0)
        total = 0.0
        for _ in range(10):
            _, r, term, trunc, info = env1.step(env1.action_space.sample())
            total += info["cash_flow"]
            if term or trunc:
                break
        assert info["cumulative_profit"] == pytest.approx(total, rel=1e-5)