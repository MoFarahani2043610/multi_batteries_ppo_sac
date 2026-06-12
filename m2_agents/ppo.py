"""
m2_agents/ppo.py
================
Milestone 2 — PPO agent wrapper around Stable-Baselines3.

Why PPO?
--------
PPO (Proximal Policy Optimization, Schulman et al. 2017) is an on-policy
algorithm that clips the policy update to stay close to the previous policy.
Key properties:
  - On-policy: collects fresh data each update, then discards it.
  - Clipped objective: prevents large destabilising policy updates.
  - Shared actor-critic: one network with separate heads for policy and value.
  - Predictable memory: no replay buffer, so memory scales linearly with n_steps.

Why it might scale better than SAC at large N:
  - No replay buffer means memory cost is fixed regardless of action dimension.
  - Clipped updates stay stable even when the policy space grows.
  - GAE advantage estimation remains well-defined in high dimensions.

Why it might be less sample efficient than SAC:
  - On-policy: each transition used once then thrown away.
  - Needs more wall-clock time to reach the same performance.

Usage
-----
    from m2_agents.ppo import PPOAgent
    agent = PPOAgent(env, config)
    agent.train()
    mean_return = agent.evaluate(n_episodes=10)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env')))

import numpy as np
import yaml
from pathlib import Path


class PPOAgent:
    """
    PPO agent using Stable-Baselines3.

    Parameters
    ----------
    env    : StorageArbitrageEnv — must be Gymnasium-compatible
    config : dict — loaded from m2_configs/ppo_n1.yaml
    seed   : int  — for full reproducibility
    """

    def __init__(self, env, config: dict, seed: int = 0):
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.monitor import Monitor
        except ImportError:
            raise ImportError(
                "stable-baselines3 not installed.\n"
                "Run: pip install stable-baselines3"
            )

        self.env     = env
        self.config  = config
        self.seed    = seed
        self.cfg     = config.get("ppo", {})
        self.log_cfg = config.get("logging", {})

        # ── log directory ─────────────────────────────────────────────
        log_dir = Path(self.log_cfg.get("log_dir", "m2_logs/ppo_n1")) / f"seed_{seed}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # ── wrap env with Monitor ─────────────────────────────────────
        from stable_baselines3.common.monitor import Monitor
        self.monitored_env = Monitor(env, str(log_dir / "monitor"))

        # ── policy kwargs ─────────────────────────────────────────────
        policy_kwargs = self.cfg.get("policy_kwargs", {})
        net_arch      = policy_kwargs.get("net_arch", {"pi": [256, 256], "vf": [256, 256]})

        # ── build model ───────────────────────────────────────────────
        tb_log = str(log_dir.parent) if self.log_cfg.get("tensorboard", True) else None

        # handle clip_range_vf: yaml null → None
        clip_range_vf = self.cfg.get("clip_range_vf", None)

        self.model = PPO(
            policy        = self.cfg.get("policy", "MlpPolicy"),
            env           = self.monitored_env,
            learning_rate = float(self.cfg.get("learning_rate", 3e-4)),
            n_steps       = int(self.cfg.get("n_steps", 2048)),
            batch_size    = int(self.cfg.get("batch_size", 64)),
            n_epochs      = int(self.cfg.get("n_epochs", 10)),
            gamma         = float(self.cfg.get("gamma", 0.99)),
            gae_lambda    = float(self.cfg.get("gae_lambda", 0.95)),
            clip_range    = float(self.cfg.get("clip_range", 0.2)),
            clip_range_vf = clip_range_vf,
            ent_coef      = float(self.cfg.get("ent_coef", 0.01)),
            vf_coef       = float(self.cfg.get("vf_coef", 0.5)),
            max_grad_norm = float(self.cfg.get("max_grad_norm", 0.5)),
            policy_kwargs = {"net_arch": net_arch},
            tensorboard_log = tb_log,
            verbose       = int(self.log_cfg.get("verbose", 1)),
            seed          = seed,
        )

        self._is_trained = False

    def train(self, total_timesteps: int = None):
        """
        Train PPO for `total_timesteps` environment steps.

        PPO-specific logged metrics (visible in TensorBoard):
          - train/approx_kl        : KL divergence between old and new policy
          - train/clip_fraction    : fraction of clipped policy updates
          - train/entropy_loss     : policy entropy
          - train/value_loss       : critic MSE loss
          - train/explained_variance : how well the value function predicts returns
          - rollout/ep_rew_mean    : mean episodic return
        """
        from stable_baselines3.common.callbacks import (
            EvalCallback, CheckpointCallback, CallbackList
        )
        from stable_baselines3.common.monitor import Monitor

        train_cfg = self.config.get("training", {})
        steps     = total_timesteps or int(train_cfg.get("total_timesteps", 500_000))
        eval_freq = int(train_cfg.get("eval_freq", 5_000))
        eval_eps  = int(train_cfg.get("eval_episodes", 10))
        save_freq = int(self.log_cfg.get("save_freq", 50_000))

        # PPO: eval_freq must be a multiple of n_steps
        n_steps   = int(self.cfg.get("n_steps", 2048))
        eval_freq = max(n_steps, (eval_freq // n_steps) * n_steps)

        eval_env = Monitor(
            self._make_fresh_env(),
            str(self.log_dir / "eval_monitor")
        )

        callbacks = CallbackList([
            EvalCallback(
                eval_env,
                best_model_save_path = str(self.log_dir / "best_model"),
                log_path             = str(self.log_dir / "eval_results"),
                eval_freq            = eval_freq,
                n_eval_episodes      = eval_eps,
                deterministic        = True,
                verbose              = 0,
            ),
            CheckpointCallback(
                save_freq   = save_freq,
                save_path   = str(self.log_dir / "checkpoints"),
                name_prefix = "ppo",
                verbose     = 0,
            ),
        ])

        print(f"\nTraining PPO  seed={self.seed}  steps={steps:,}")
        print(f"  Log dir : {self.log_dir}")
        print(f"  TensorBoard: tensorboard --logdir {self.log_dir.parent}\n")

        self.model.learn(
            total_timesteps     = steps,
            callback            = callbacks,
            log_interval        = int(self.config.get("training", {}).get("log_interval", 1)),
            tb_log_name         = f"PPO_seed{self.seed}",
            reset_num_timesteps = True,
        )

        self._is_trained = True
        self.model.save(str(self.log_dir / "final_model"))
        print(f"\nPPO training complete. Model saved to {self.log_dir}/final_model")

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> dict:
        """
        Evaluate the trained agent on fresh episodes.

        Returns
        -------
        dict: mean_return, std_return, min_return, max_return, returns
        """
        env     = self._make_fresh_env()
        returns = []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=1000 + ep)
            done      = False
            ep_return = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                done       = terminated or truncated
            returns.append(ep_return)

        returns = np.array(returns)
        return {
            "mean_return": float(returns.mean()),
            "std_return":  float(returns.std()),
            "min_return":  float(returns.min()),
            "max_return":  float(returns.max()),
            "returns":     returns.tolist(),
        }

    def load(self, path: str):
        """Load a saved model from path."""
        from stable_baselines3 import PPO
        self.model       = PPO.load(path, env=self.monitored_env)
        self._is_trained = True
        print(f"Loaded PPO model from {path}")

    def _make_fresh_env(self):
        """Create a fresh env instance with the same config and price source."""
        from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource, SyntheticPriceSource
        env_cfg      = self.config.get("env", {})
        price_source = env_cfg.get("price_source", "synthetic").lower()

        if price_source == "caiso":
            import sys, os
            _ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
            _DATA = os.path.join(_ROOT, 'data')
            if _DATA not in sys.path:
                sys.path.insert(0, _DATA)
            from loader import load_prices, make_features
            prices, timestamps = load_prices(
                market    = "caiso",
                cache_dir = os.path.join(_ROOT, "cache"),
            )
            features = make_features(timestamps)
            source   = HistoricalPriceSource(
                prices      = prices,
                features    = features,
                episode_len = 288,
            )
        else:
            source = SyntheticPriceSource()

        return StorageArbitrageEnv(
            n_batteries         = int(env_cfg.get("n_batteries", 1)),
            dt_hours            = float(env_cfg.get("dt_hours", 5/60)),
            degradation_penalty = float(env_cfg.get("degradation_penalty", 0.0)),
            normalize_obs       = bool(env_cfg.get("normalize_obs", True)),
            price_ref           = float(env_cfg.get("price_ref", 52.0)),
            price_source        = source,
        )


# ─────────────────────────────────────────────────────
#  Convenience loader
# ─────────────────────────────────────────────────────

def load_ppo_config(path: str = "m2_configs/ppo_n1.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_ppo_agent(config_path: str, seed: int = 0) -> PPOAgent:
    from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource, SyntheticPriceSource
    config       = load_ppo_config(config_path)
    env_cfg      = config.get("env", {})
    price_source = env_cfg.get("price_source", "synthetic").lower()

    if price_source == "caiso":
        import os
        _ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        import sys
        _DATA = os.path.join(_ROOT, 'data')
        if _DATA not in sys.path:
            sys.path.insert(0, _DATA)
        from loader import load_prices, make_features
        prices, timestamps = load_prices(market="caiso", cache_dir=os.path.join(_ROOT, "cache"))
        features = make_features(timestamps)
        source   = HistoricalPriceSource(prices=prices, features=features, episode_len=288)
    else:
        source = SyntheticPriceSource()

    env = StorageArbitrageEnv(
        n_batteries         = int(env_cfg.get("n_batteries", 1)),
        dt_hours            = float(env_cfg.get("dt_hours", 5/60)),
        degradation_penalty = float(env_cfg.get("degradation_penalty", 0.0)),
        normalize_obs       = bool(env_cfg.get("normalize_obs", True)),
        price_ref           = float(env_cfg.get("price_ref", 52.0)),
        price_source        = source,
    )
    return PPOAgent(env, config, seed=seed)