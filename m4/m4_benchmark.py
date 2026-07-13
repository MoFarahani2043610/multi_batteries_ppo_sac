import sys, time
sys.path.insert(0, 'env')
from storage_arbitrage_env import StorageArbitrageEnv
from stable_baselines3 import SAC, PPO

env = StorageArbitrageEnv(n_batteries=10)

# Benchmark SAC
model = SAC('MlpPolicy', env, verbose=0, learning_starts=1000)
start = time.time()
model.learn(total_timesteps=10000)
elapsed = time.time() - start
print(f'SAC 10K steps: {elapsed:.1f}s  ->  1M steps estimate: {elapsed*100/3600:.1f} hours')

# Benchmark PPO
model2 = PPO('MlpPolicy', env, verbose=0)
start = time.time()
model2.learn(total_timesteps=10000)
elapsed2 = time.time() - start
print(f'PPO 10K steps: {elapsed2:.1f}s  ->  1M steps estimate: {elapsed2*100/3600:.1f} hours')