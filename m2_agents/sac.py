"""
m2_agents/sac.py
================
Milestone 2 — SAC agent wrapper around Stable-Baselines3.

Documents every design choice so the thesis can reference this file directly.

Why SAC?
--------
SAC (Soft Actor-Critic, Haarnoja et al. 2018) is an off-policy algorithm that
maximises a trade-off between reward and entropy. Key properties:
  - Off-policy: learns from a replay buffer, so each transition is reused many times.
  - Automatic entropy tuning: the temperature parameter α is adjusted automatically
    so the policy maintains a target level of randomness.
  - Continuous actions: the policy outputs a Gaussian distribution over R^N,
    exactly matching our action space [−P_dis, +P_chg]^N.
  - Sample efficient: typically converges faster than PPO in low-data regimes.

Why it might struggle at scale (N→20):
  - The critic must learn Q(s,a) over a 20-dimensional action space.
  - The replay buffer grows with experience but memory is fixed.
  - Entropy regularisation may become harder to tune as action dim grows.

Learning rate schedule (added July 2026): a constant learning rate was found
to produce oscillation / non-improvement in the second half of training
(after ~150-300k of 1M steps). config.sac.lr_schedule: "linear" enables a
linear decay from the configured learning_rate down to ~0 over training,
which is standard practice for this failure mode. Default remains constant
if lr_schedule is unset, preserving prior behavior.

Usage
-----
    from m2_agents.sac import SACAgent
    agent = SACAgent(env, config)
    agent.train()
    mean_return = agent.evaluate(n_episodes=10)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env')))

import numpy as np
import yaml
from pathlib import Path


class SACAgent:
    """
    SAC agent using Stable-Baselines3.

    Parameters
    ----------
    env    : StorageArbitrageEnv — must be Gymnasium-compatible
    config : dict — loaded from m2_configs/sac_n1.yaml
    seed   : int  — for full reproducibility
    """

    def __init__(self, env, config: dict, seed: int = 0):
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import (
                EvalCallback, CheckpointCallback, CallbackList
            )
            from stable_baselines3.common.monitor import Monitor
        except ImportError:
            raise ImportError(
                "stable-baselines3 not installed.\n"
                "Run: pip install stable-baselines3"
            )

        self.env    = env
        self.config = config
        self.seed   = seed
        self.cfg    = config.get("sac", {})
        self.log_cfg = config.get("logging", {})

        # ── log directory ────────────────────────────────────────────
        log_dir = Path(self.log_cfg.get("log_dir", "m2_logs/sac_n1")) / f"seed_{seed}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # ── wrap env with Monitor for episode logging ─────────────────
        # Monitor records episodic return and length automatically
        self.monitored_env = Monitor(env, str(log_dir / "monitor"))

        # ── policy kwargs ─────────────────────────────────────────────
        policy_kwargs = self.cfg.get("policy_kwargs", {})
        net_arch = policy_kwargs.get("net_arch", [256, 256])

        # ── build model ───────────────────────────────────────────────
        tb_log = str(log_dir.parent) if self.log_cfg.get("tensorboard", True) else None

        # ── learning rate: fixed float or linear decay schedule ────────
        lr_value = float(self.cfg.get("learning_rate", 3e-4))
        lr_schedule = self.cfg.get("lr_schedule", "constant")
        if lr_schedule == "linear":
            # Linearly decay from lr_value down to ~0 over training.
            # SB3 calls this with progress_remaining going from 1.0 (start)
            # to 0.0 (end of training).
            def _linear_lr(progress_remaining: float, _lr0: float = lr_value):
                return progress_remaining * _lr0
            learning_rate_arg = _linear_lr
        else:
            learning_rate_arg = lr_value

        self.model = SAC(
            policy         = self.cfg.get("policy", "MlpPolicy"),
            env            = self.monitored_env,
            learning_rate  = learning_rate_arg,
            buffer_size    = int(self.cfg.get("buffer_size", 100_000)),
            learning_starts= int(self.cfg.get("learning_starts", 5_000)),
            batch_size     = int(self.cfg.get("batch_size", 256)),
            tau            = float(self.cfg.get("tau", 0.005)),
            gamma          = float(self.cfg.get("gamma", 0.99)),
            train_freq     = int(self.cfg.get("train_freq", 1)),
            gradient_steps = int(self.cfg.get("gradient_steps", 1)),
            ent_coef       = self.cfg.get("ent_coef", "auto"),
            target_entropy = self.cfg.get("target_entropy", "auto"),
            policy_kwargs  = {"net_arch": net_arch},
            tensorboard_log= tb_log,
            verbose        = int(self.log_cfg.get("verbose", 1)),
            seed           = seed,
        )

        self._is_trained = False

    def train(self, total_timesteps: int = None):
        """
        Train SAC for `total_timesteps` environment steps.

        Sets up:
          - EvalCallback: runs evaluation every eval_freq steps and saves best model
          - CheckpointCallback: saves model weights every save_freq steps
          - TensorBoard logging: critic_loss, actor_loss, ent_coef, entropy

        All logged to self.log_dir / TensorBoard.
        """
        from stable_baselines3.common.callbacks import (
            EvalCallback, CheckpointCallback, CallbackList
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.env_util import make_vec_env

        train_cfg = self.config.get("training", {})
        steps     = total_timesteps or int(train_cfg.get("total_timesteps", 500_000))
        eval_freq = int(train_cfg.get("eval_freq", 5_000))
        eval_eps  = int(train_cfg.get("eval_episodes", 10))
        save_freq = int(self.log_cfg.get("save_freq", 50_000))

        # separate eval env (same config, different seed to avoid data leakage)
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
                save_freq = save_freq,
                save_path = str(self.log_dir / "checkpoints"),
                name_prefix = "sac",
                verbose   = 0,
            ),
        ])

        print(f"\nTraining SAC  seed={self.seed}  steps={steps:,}")
        print(f"  Log dir : {self.log_dir}")
        print(f"  TensorBoard: tensorboard --logdir {self.log_dir.parent}\n")

        self.model.learn(
            total_timesteps  = steps,
            callback         = callbacks,
            log_interval     = int(self.config.get("training", {}).get("log_interval", 1)),
            tb_log_name      = f"SAC_seed{self.seed}",
            reset_num_timesteps = True,
        )

        self._is_trained = True
        self.model.save(str(self.log_dir / "final_model"))
        print(f"\nSAC training complete. Model saved to {self.log_dir}/final_model")

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> dict:
        """
        Evaluate the trained agent on fresh episodes.

        Parameters
        ----------
        n_episodes    : number of evaluation episodes
        deterministic : use deterministic policy (recommended for evaluation)

        Returns
        -------
        dict with keys: mean_return, std_return, mean_steps, returns
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
        from stable_baselines3 import SAC
        self.model       = SAC.load(path, env=self.monitored_env)
        self._is_trained = True
        print(f"Loaded SAC model from {path}")

    def _make_fresh_env(self):
        """Create a fresh env instance with the same config and price source.

        Always uses real CAISO data. There is no synthetic-data fallback:
        config.env.price_source must be explicitly "caiso", or this raises
        an error rather than silently substituting synthetic data.
        """
        from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
        env_cfg      = self.config.get("env", {})
        price_source = env_cfg.get("price_source", "").lower()

        if price_source != "caiso":
            raise ValueError(
                f"config.env.price_source must be 'caiso', got {price_source!r}. "
                "Synthetic data is no longer supported — set price_source: caiso "
                "in the YAML config."
            )

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

def load_sac_config(path: str = "m2_configs/sac_n1.yaml") -> dict:
    """Load a YAML config file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def make_sac_agent(config_path: str, seed: int = 0) -> SACAgent:
    """One-liner: load config + build agent.

    Always uses real CAISO data. There is no synthetic-data fallback:
    config.env.price_source must be explicitly "caiso", or this raises
    an error rather than silently substituting synthetic data.
    """
    from storage_arbitrage_env import StorageArbitrageEnv, HistoricalPriceSource
    config       = load_sac_config(config_path)
    env_cfg      = config.get("env", {})
    price_source = env_cfg.get("price_source", "").lower()

    if price_source != "caiso":
        raise ValueError(
            f"config.env.price_source must be 'caiso', got {price_source!r}. "
            "Synthetic data is no longer supported — set price_source: caiso "
            "in the YAML config."
        )

    import os, sys
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    _DATA = os.path.join(_ROOT, 'data')
    if _DATA not in sys.path:
        sys.path.insert(0, _DATA)
    from loader import load_prices, make_features
    prices, timestamps = load_prices(market="caiso", cache_dir=os.path.join(_ROOT, "cache"))
    features = make_features(timestamps)
    source   = HistoricalPriceSource(prices=prices, features=features, episode_len=288)

    env = StorageArbitrageEnv(
        n_batteries         = int(env_cfg.get("n_batteries", 1)),
        dt_hours            = float(env_cfg.get("dt_hours", 5/60)),
        degradation_penalty = float(env_cfg.get("degradation_penalty", 0.0)),
        normalize_obs       = bool(env_cfg.get("normalize_obs", True)),
        price_ref           = float(env_cfg.get("price_ref", 52.0)),
        price_source        = source,
    )
    return SACAgent(env, config, seed=seed)