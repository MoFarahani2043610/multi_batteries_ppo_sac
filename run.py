import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines.perfect_foresight import compare_all
from storage_arbitrage_env import StorageArbitrageEnv

env = StorageArbitrageEnv(n_batteries=1)
compare_all(env, n_episodes=5)