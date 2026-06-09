"""SAC agent for Milestone 2"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

import numpy as np
import yaml
from pathlib import Path

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
except ImportError:
    raise ImportError("pip install stable-baselines3")


class SACAgent:
    def __init__(self, env, config: dict, seed: int = 0):
        self.env = env
        self.config = config
        self.seed = seed
        
        cfg = config.get("sac", {})
        log_cfg = config.get("logging", {})
        
        log_dir = Path(log_cfg.get("log_dir", "m2_logs/sac_n1")) / f"seed_{seed}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        
        self.monitored_env = Monitor(env, str(log_dir / "monitor"))
        
        policy_kwargs = cfg.get("policy_kwargs", {})
        net_arch = policy_kwargs.get("net_arch", [256, 256])
        
        tb_log = str(log_dir.parent) if log_cfg.get("tensorboard", True) else None
        
        self.model = SAC(
            policy=cfg.get("policy", "MlpPolicy"),
            env=self.monitored_env,
            learning_rate=float(cfg.get("learning_rate", 3e-4)),
            buffer_size=int(cfg.get("buffer_size", 100000)),
            learning_starts=int(cfg.get("learning_starts", 5000)),
            batch_size=int(cfg.get("batch_size", 256)),
            tau=float(cfg.get("tau", 0.005)),
            gamma=float(cfg.get("gamma", 0.99)),
            train_freq=int(cfg.get("train_freq", 1)),
            gradient_steps=int(cfg.get("gradient_steps", 1)),
            ent_coef=cfg.get("ent_coef", "auto"),
            target_entropy=cfg.get("target_entropy", "auto"),
            policy_kwargs={"net_arch": net_arch},
            tensorboard_log=tb_log,
            verbose=int(log_cfg.get("verbose", 1)),
            seed=seed,
        )
        self._is_trained = False
    
    def train(self, total_timesteps: int = None):
        train_cfg = self.config.get("training", {})
        steps = total_timesteps or int(train_cfg.get("total_timesteps", 500000))
        eval_freq = int(train_cfg.get("eval_freq", 5000))
        eval_eps = int(train_cfg.get("eval_episodes", 10))
        save_freq = int(self.config.get("logging", {}).get("save_freq", 50000))
        
        eval_env = Monitor(self._make_fresh_env(), str(self.log_dir / "eval_monitor"))
        
        callbacks = CallbackList([
            EvalCallback(
                eval_env,
                best_model_save_path=str(self.log_dir / "best_model"),
                log_path=str(self.log_dir / "eval_results"),
                eval_freq=eval_freq,
                n_eval_episodes=eval_eps,
                deterministic=True,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(self.log_dir / "checkpoints"),
                name_prefix="sac",
                verbose=0,
            ),
        ])
        
        print(f"\nTraining SAC  seed={self.seed}  steps={steps:,}")
        print(f"  Log dir: {self.log_dir}\n")
        
        self.model.learn(
            total_timesteps=steps,
            callback=callbacks,
            log_interval=int(self.config.get("training", {}).get("log_interval", 1)),
            tb_log_name=f"SAC_seed{self.seed}",
        )
        
        self._is_trained = True
        self.model.save(str(self.log_dir / "final_model"))
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> dict:
        env = self._make_fresh_env()
        returns = []
        
        for ep in range(n_episodes):
            obs, info = env.reset(seed=1000 + ep)
            done = False
            ep_return = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                done = terminated or truncated
            returns.append(ep_return)
        
        returns = np.array(returns)
        return {
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "returns": returns.tolist(),
        }
    
    def _make_fresh_env(self):
        from storage_arbitrage_env import StorageArbitrageEnv
        env_cfg = self.config.get("env", {})
        return StorageArbitrageEnv(
            n_batteries=int(env_cfg.get("n_batteries", 1)),
            dt_hours=float(env_cfg.get("dt_hours", 5/60)),
            degradation_penalty=float(env_cfg.get("degradation_penalty", 0.0)),
            normalize_obs=bool(env_cfg.get("normalize_obs", True)),
            price_ref=float(env_cfg.get("price_ref", 50.0)),
        )