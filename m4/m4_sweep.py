"""
Milestone 4 — Sample Efficiency vs Wall-Clock Tradeoff
Sweep parallelism/data-reuse axis at N=10, fixed 1M steps.
Hardware: Intel i7 (4 physical / 8 logical cores), 17GB RAM, no GPU.
"""

import os, sys, time, json, yaml
import numpy as np
sys.path.insert(0, 'env')
sys.path.insert(0, 'm2_agents')

from storage_arbitrage_env import StorageArbitrageEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# ── Config ────────────────────────────────────────────────────────
N_BATTERIES     = 10
TOTAL_STEPS     = 1_000_000
EVAL_EPISODES   = 5
EVAL_FREQ       = 50_000   # evaluate every 50K steps
LOG_DIR         = "m4_logs"
RESULTS_DIR     = "experiments/m4_throughput"
SEED            = 0

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hardware info ─────────────────────────────────────────────────
HARDWARE = {
    "gpu": "None",
    "cpu": "Intel64 Family 6 Model 142 (4 physical / 8 logical cores)",
    "ram_gb": 17.0,
    "note": "All runs on CPU only. Wall-clock times are CPU-bound."
}

# ── Sweep configurations ──────────────────────────────────────────
SAC_CONFIGS = [
    {"utd": 1,  "batch_size": 256},
    {"utd": 1,  "batch_size": 512},
    {"utd": 1,  "batch_size": 1024},
    {"utd": 4,  "batch_size": 256},
    {"utd": 4,  "batch_size": 512},
    {"utd": 4,  "batch_size": 1024},
    {"utd": 8,  "batch_size": 256},
    {"utd": 8,  "batch_size": 512},
    {"utd": 8,  "batch_size": 1024},
    {"utd": 16, "batch_size": 256},
    {"utd": 16, "batch_size": 512},
    {"utd": 16, "batch_size": 1024},
]

PPO_CONFIGS = [
    {"n_envs": 1},
    {"n_envs": 4},
    {"n_envs": 16},
    {"n_envs": 64},
]

# ── Evaluation callback ───────────────────────────────────────────
class WallClockCallback(BaseCallback):
    """Records (steps, wall_clock_seconds, mean_return) at each eval point."""
    def __init__(self, eval_env, eval_episodes, eval_freq, start_time):
        super().__init__()
        self.eval_env      = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq     = eval_freq
        self.start_time    = start_time
        self.records       = []  # list of (steps, seconds, mean_return)

    def _on_step(self):
        if self.num_timesteps >= len(self.records) * self.eval_freq + self.eval_freq:
            returns = []
            for _ in range(self.eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_return = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    ep_return += reward
                    done = terminated or truncated
                returns.append(ep_return)
            mean_return  = float(np.mean(returns))
            wall_seconds = time.time() - self.start_time
            steps        = self.num_timesteps
            self.records.append((steps, wall_seconds, mean_return))
            print(f"  step={steps:>8,}  wall={wall_seconds/60:>6.1f}min  return=${mean_return:>8.2f}")
        return True

# ── Run one SAC config ────────────────────────────────────────────
def run_sac(utd, batch_size):
    run_name = f"sac_utd{utd}_bs{batch_size}"
    out_path = os.path.join(LOG_DIR, run_name)
    result_path = os.path.join(RESULTS_DIR, f"{run_name}.json")

    if os.path.exists(result_path):
        print(f"[SKIP] {run_name} already done")
        return

    print(f"\n{'='*60}")
    print(f"SAC  UTD={utd}  batch_size={batch_size}")
    print(f"{'='*60}")

    os.makedirs(out_path, exist_ok=True)
    env      = StorageArbitrageEnv(n_batteries=N_BATTERIES)
    eval_env = StorageArbitrageEnv(n_batteries=N_BATTERIES)

    model = SAC(
        'MlpPolicy', env,
        learning_rate        = 1e-3,
        buffer_size          = 500_000,
        learning_starts      = 1_000,
        batch_size           = batch_size,
        train_freq           = 1,
        gradient_steps       = utd,
        policy_kwargs        = dict(net_arch=[400, 300]),
        ent_coef             = 'auto',
        target_entropy       = 'auto',
        tensorboard_log      = out_path,
        verbose              = 0,
        seed                 = SEED,
    )

    start_time = time.time()
    cb = WallClockCallback(eval_env, EVAL_EPISODES, EVAL_FREQ, start_time)
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)
    total_wall = time.time() - start_time

    result = {
        "algo": "SAC", "utd": utd, "batch_size": batch_size,
        "n_batteries": N_BATTERIES, "total_steps": TOTAL_STEPS,
        "total_wall_seconds": total_wall,
        "hardware": HARDWARE,
        "records": cb.records,   # [(steps, seconds, mean_return), ...]
        "final_return": cb.records[-1][2] if cb.records else None,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    fr = result["final_return"]
    print(f"Done. Wall time: {total_wall/3600:.2f}h  Final return: ${fr:.2f}" if fr is not None else f"Done. Wall time: {total_wall/3600:.2f}h  Final return: N/A")

# ── Run one PPO config ────────────────────────────────────────────
def run_ppo(n_envs):
    run_name = f"ppo_nenvs{n_envs}"
    out_path = os.path.join(LOG_DIR, run_name)
    result_path = os.path.join(RESULTS_DIR, f"{run_name}.json")

    if os.path.exists(result_path):
        print(f"[SKIP] {run_name} already done")
        return

    print(f"\n{'='*60}")
    print(f"PPO  n_envs={n_envs}")
    print(f"{'='*60}")

    os.makedirs(out_path, exist_ok=True)
    eval_env = StorageArbitrageEnv(n_batteries=N_BATTERIES)

    def make_env():
        return StorageArbitrageEnv(n_batteries=N_BATTERIES)

    vec_env = make_vec_env(make_env, n_envs=n_envs)

    model = PPO(
        'MlpPolicy', vec_env,
        learning_rate  = 1e-3,
        n_steps        = max(2048 // n_envs, 64),
        batch_size     = 64,
        ent_coef       = 0.001,
        policy_kwargs  = dict(net_arch=[400, 300]),
        tensorboard_log= out_path,
        verbose        = 0,
        seed           = SEED,
    )

    start_time = time.time()
    cb = WallClockCallback(eval_env, EVAL_EPISODES, EVAL_FREQ, start_time)
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)
    total_wall = time.time() - start_time

    result = {
        "algo": "PPO", "n_envs": n_envs,
        "n_batteries": N_BATTERIES, "total_steps": TOTAL_STEPS,
        "total_wall_seconds": total_wall,
        "hardware": HARDWARE,
        "records": cb.records,
        "final_return": cb.records[-1][2] if cb.records else None,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    fr = result["final_return"]
    print(f"Done. Wall time: {total_wall/3600:.2f}h  Final return: ${fr:.2f}" if fr is not None else f"Done. Wall time: {total_wall/3600:.2f}h  Final return: N/A")

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',  choices=['sac','ppo','all'], default='all')
    parser.add_argument('--utd',   type=int, default=None, help='SAC UTD ratio (run single config)')
    parser.add_argument('--bs',    type=int, default=None, help='SAC batch size (run single config)')
    parser.add_argument('--nenvs', type=int, default=None, help='PPO n_envs (run single config)')
    args = parser.parse_args()

    print(f"Hardware: {HARDWARE['cpu']}, RAM={HARDWARE['ram_gb']}GB, GPU={HARDWARE['gpu']}")
    print(f"N={N_BATTERIES}, total_steps={TOTAL_STEPS:,}, eval_freq={EVAL_FREQ:,}")

    if args.utd and args.bs:
        run_sac(args.utd, args.bs)
    elif args.nenvs:
        run_ppo(args.nenvs)
    elif args.algo == 'sac':
        for cfg in SAC_CONFIGS:
            run_sac(cfg['utd'], cfg['batch_size'])
    elif args.algo == 'ppo':
        for cfg in PPO_CONFIGS:
            run_ppo(cfg['n_envs'])
    else:
        for cfg in SAC_CONFIGS:
            run_sac(cfg['utd'], cfg['batch_size'])
        for cfg in PPO_CONFIGS:
            run_ppo(cfg['n_envs'])

